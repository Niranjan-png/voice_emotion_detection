import numpy as np
import pyaudio
import threading
import queue
import time
import librosa
from utils.audio_utils import normalize_audio, bandpass_filter
from feature_extractor import FeatureExtractor

class AudioProcessor:
    """
    Class for real-time audio capture and processing
    """
    def __init__(self, 
                 rate=16000, 
                 chunk_size=1024, 
                 channels=1, 
                 format_type=pyaudio.paFloat32,
                 window_size=2,  # window size in seconds
                 hop_size=0.5,   # hop size in seconds
                 feature_extractor=None):
        """Initialize the audio processor"""
        self.rate = rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format_type
        
        # Calculate window and hop size in samples
        self.window_samples = int(window_size * rate)
        self.hop_samples = int(hop_size * rate)
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Initialize feature extractor if not provided
        if feature_extractor is None:
            self.feature_extractor = FeatureExtractor(
                sample_rate=rate,
                duration=window_size
            )
        else:
            self.feature_extractor = feature_extractor
        
        # Buffer for storing audio data
        self.buffer = np.zeros(self.window_samples, dtype=np.float32)
        
        # Queue for feature extraction and prediction
        self.feature_queue = queue.Queue()
        
        # Flags for control
        self.is_recording = False
        self.stream = None
        self.processing_thread = None
    
    def start_recording(self):
        """Start recording audio from microphone"""
        if self.is_recording:
            print("Already recording")
            return False
        
        try:
            self.stream = self.audio.open(
                format=self.format, 
                channels=self.channels,
                rate=self.rate, 
                input=True,
                frames_per_buffer=self.chunk_size, 
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            self.processing_thread = threading.Thread(target=self._process_audio)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            print("Recording started")
            return True
        except Exception as e:
            print(f"Error starting recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop recording audio"""
        if not self.is_recording:
            print("Not recording")
            return False
        
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # Wait for processing thread to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None
        
        print("Recording stopped")
        return True
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio input"""
        if not self.is_recording:
            return (None, pyaudio.paComplete)
        
        # Convert byte data to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to buffer (shift buffer and add new data)
        self.buffer = np.roll(self.buffer, -len(audio_data))
        self.buffer[-len(audio_data):] = audio_data
        
        # Add buffer to queue for processing
        self.feature_queue.put(np.copy(self.buffer))
        
        return (None, pyaudio.paContinue)
    
    def _process_audio(self):
        """Process audio data from the queue"""
        last_process_time = 0
        min_process_interval = self.hop_samples / self.rate  # minimum time between processing in seconds
        
        while self.is_recording:
            # Get buffer from queue
            try:
                # Non-blocking get to allow checking is_recording flag
                buffer = self.feature_queue.get(block=True, timeout=0.1)
                
                # Ensure minimum time between processing
                current_time = time.time()
                if current_time - last_process_time >= min_process_interval:
                    # Preprocess audio
                    preprocessed = self._preprocess_buffer(buffer)
                    
                    # Extract features
                    features = self.feature_extractor.extract_features(preprocessed, self.rate)
                    
                    # Call the on_features callback if defined
                    if hasattr(self, 'on_features') and callable(self.on_features):
                        self.on_features(features)
                    
                    last_process_time = current_time
                
                self.feature_queue.task_done()
            except queue.Empty:
                # Queue is empty, continue
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
    
    def _preprocess_buffer(self, buffer):
        """Preprocess audio buffer"""
        # Normalize
        audio = normalize_audio(buffer)
        
        # Apply bandpass filter (focus on speech: 80Hz-8000Hz)
        audio = bandpass_filter(audio, 80, 8000, self.rate)
        
        return audio
    
    def set_on_features_callback(self, callback):
        """Set callback function for feature extraction"""
        self.on_features = callback
    
    def __del__(self):
        """Cleanup resources"""
        self.stop_recording()
        if self.audio:
            self.audio.terminate()
