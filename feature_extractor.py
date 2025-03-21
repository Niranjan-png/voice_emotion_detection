import numpy as np
import librosa
import librosa.display
import torch
from tqdm import tqdm
import os
from utils.audio_utils import normalize_audio, bandpass_filter

class FeatureExtractor:
    """
    Class for extracting audio features for emotion recognition
    """
    
    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128, 
                 n_mfcc=40, frame_length=None, duration=None):
        """
        Initialize the feature extractor
        
        Parameters:
        ----------
        sample_rate : int
            Audio sample rate
        n_fft : int
            FFT window size
        hop_length : int
            Hop length for STFT
        n_mels : int
            Number of Mel bands
        n_mfcc : int
            Number of MFCCs to extract
        frame_length : int
            Frame length for frame-level features (in samples)
        duration : float
            Duration to trim audio to (in seconds), if None, uses the entire audio
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length
        self.duration = duration
        
    def preprocess_audio(self, audio_data, sr=None):
        """
        Preprocess audio data before feature extraction
        
        Parameters:
        ----------
        audio_data : np.ndarray
            Audio time series
        sr : int
            Sample rate of the audio
            
        Returns:
        -------
        processed_audio : np.ndarray
            Preprocessed audio
        """
        # Use passed sample rate or default
        sr = sr or self.sample_rate
        
        # Resample if necessary
        if sr != self.sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
            sr = self.sample_rate
            
        # Normalize audio
        audio_data = normalize_audio(audio_data)
        
        # Apply bandpass filter to focus on speech frequency range (80Hz-8000Hz)
        audio_data = bandpass_filter(audio_data, 80, 8000, sr)
        
        # Trim to specified duration if provided
        if self.duration is not None:
            max_samples = int(self.duration * sr)
            if len(audio_data) > max_samples:
                audio_data = audio_data[:max_samples]
            elif len(audio_data) < max_samples:
                # Pad with zeros if too short
                padding = max_samples - len(audio_data)
                audio_data = np.pad(audio_data, (0, padding), mode='constant')
        
        return audio_data
    
    def extract_features(self, audio_data, sr=None, return_dict=False):
        """
        Extract acoustic features from audio data
        
        Parameters:
        ----------
        audio_data : np.ndarray
            Audio time series
        sr : int
            Sample rate
        return_dict : bool
            If True, return features as a dictionary, otherwise as a flattened array
            
        Returns:
        -------
        features : np.ndarray or dict
            Extracted features
        """
        # Preprocess the audio
        audio_data = self.preprocess_audio(audio_data, sr)
        sr = self.sample_rate  # After preprocessing, sr is self.sample_rate
        
        features = {}
        
        # Extract Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features['mel_spectrogram'] = mel_spec_db
        
        # Extract MFCCs and their deltas
        mfccs = librosa.feature.mfcc(
            y=audio_data, 
            sr=sr, 
            n_mfcc=self.n_mfcc, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        # Compute deltas and delta-deltas (1st and 2nd derivatives)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        features['mfcc'] = mfccs
        features['mfcc_delta'] = mfcc_delta
        features['mfcc_delta2'] = mfcc_delta2
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_data, 
            sr=sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio_data, 
            sr=sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio_data, 
            sr=sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_data, 
            sr=sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        features['spectral_centroid'] = spectral_centroid
        features['spectral_bandwidth'] = spectral_bandwidth
        features['spectral_contrast'] = spectral_contrast
        features['spectral_rolloff'] = spectral_rolloff
        
        # Extract time-domain features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            audio_data, 
            frame_length=self.n_fft, 
            hop_length=self.hop_length
        )
        
        features['zero_crossing_rate'] = zero_crossing_rate
        
        # Extract RMS energy
        rms = librosa.feature.rms(
            y=audio_data, 
            frame_length=self.n_fft, 
            hop_length=self.hop_length
        )
        
        features['rms_energy'] = rms
        
        # Fundamental frequency (F0) using harmonic model
        if len(audio_data) > self.n_fft:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_data, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            features['f0'] = f0
            features['voiced_flag'] = voiced_flag
            features['voiced_probs'] = voiced_probs
        
        if return_dict:
            return features
        else:
            # Flatten and concatenate all features
            # First we'll aggregate each feature along time
            agg_features = []
            
            # For MFCCs and spectral features, compute statistics
            for feature_name in ['mfcc', 'mfcc_delta', 'mfcc_delta2', 
                                'spectral_centroid', 'spectral_bandwidth', 
                                'spectral_contrast', 'spectral_rolloff', 
                                'zero_crossing_rate', 'rms_energy']:
                if feature_name in features:
                    feat = features[feature_name]
                    # Compute mean and std across time
                    feat_mean = np.mean(feat, axis=1)
                    feat_std = np.std(feat, axis=1)
                    agg_features.extend([feat_mean, feat_std])
            
            # For F0, compute additional statistics if available
            if 'f0' in features and features['f0'] is not None:
                f0 = features['f0']
                f0 = f0[~np.isnan(f0)]  # Remove NaN values
                if len(f0) > 0:
                    f0_mean = np.mean(f0)
                    f0_std = np.std(f0)
                    f0_min = np.min(f0)
                    f0_max = np.max(f0)
                    agg_features.extend([[f0_mean, f0_std, f0_min, f0_max]])
            
            # Flatten the list of features
            flat_features = np.concatenate([feat.flatten() for feat in agg_features])
            
            return flat_features
    
    def extract_features_from_file(self, file_path, return_dict=False):
        """
        Extract features from an audio file
        
        Parameters:
        ----------
        file_path : str
            Path to the audio file
        return_dict : bool
            If True, return features as a dictionary, otherwise as a flattened array
            
        Returns:
        -------
        features : np.ndarray or dict
            Extracted features
        """
        try:
            audio_data, sr = librosa.load(file_path, sr=self.sample_rate)
            return self.extract_features(audio_data, sr, return_dict)
        except Exception as e:
            print(f"Error extracting features from {file_path}: {e}")
            return None
    
    def extract_batch_features(self, audio_files, num_workers=1, return_dict=False):
        """
        Extract features from a batch of audio files
        
        Parameters:
        ----------
        audio_files : list
            List of audio file paths
        num_workers : int
            Number of parallel workers
        return_dict : bool
            If True, return features as dictionaries, otherwise as flattened arrays
            
        Returns:
        -------
        batch_features : list
            List of extracted features
        """
        batch_features = []
        
        for file_path in tqdm(audio_files, desc="Extracting features"):
            features = self.extract_features_from_file(file_path, return_dict)
            if features is not None:
                batch_features.append(features)
        
        return batch_features
    
    def extract_features_to_dataset(self, audio_files, labels, output_file, num_workers=1):
        """
        Extract features from audio files and save as a dataset
        
        Parameters:
        ----------
        audio_files : list
            List of audio file paths
        labels : list
            List of labels corresponding to audio files
        output_file : str
            Path to save the dataset
        num_workers : int
            Number of parallel workers
            
        Returns:
        -------
        bool
            True if successful, False otherwise
        """
        try:
            features = self.extract_batch_features(audio_files, num_workers, return_dict=False)
            
            # Save the dataset
            dataset = {
                'features': np.array(features),
                'labels': np.array(labels)
            }
            
            torch.save(dataset, output_file)
            print(f"Dataset saved to {output_file}")
            return True
        except Exception as e:
            print(f"Error creating dataset: {e}")
            return False
    
    def extract_spectrograms_to_folder(self, audio_files, labels, output_folder, file_prefix="spec"):
        """
        Extract mel spectrograms from audio files and save as individual files
        
        Parameters:
        ----------
        audio_files : list
            List of audio file paths
        labels : list
            List of labels corresponding to audio files
        output_folder : str
            Folder to save the spectrograms
        file_prefix : str
            Prefix for the saved files
            
        Returns:
        -------
        bool
            True if successful, False otherwise
        """
        try:
            os.makedirs(output_folder, exist_ok=True)
            
            label_map = {}
            
            for i, (file_path, label) in enumerate(zip(audio_files, labels)):
                features = self.extract_features_from_file(file_path, return_dict=True)
                
                if features is not None and 'mel_spectrogram' in features:
                    spec = features['mel_spectrogram']
                    
                    # Save the spectrogram
                    filename = f"{file_prefix}_{i}.pt"
                    filepath = os.path.join(output_folder, filename)
                    torch.save(spec, filepath)
                    
                    # Add to label map
                    label_map[filename] = label
            
            # Save the label map
            torch.save(label_map, os.path.join(output_folder, "label_map.pt"))
            
            print(f"Spectrograms saved to {output_folder}")
            return True
        except Exception as e:
            print(f"Error saving spectrograms: {e}")
            return False 
