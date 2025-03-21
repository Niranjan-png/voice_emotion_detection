import numpy as np
import pyaudio
import threading
import time
import soundfile as sf

class AudioRecorder:
    """
    Class for recording audio to file
    """
    def __init__(self, 
                 rate=16000, 
                 chunk_size=1024, 
                 channels=1, 
                 format_type=pyaudio.paFloat32):
        """
        Initialize the audio recorder
        
        Parameters:
        ----------
        rate : int
            Sample rate
        chunk_size : int
            Number of frames per buffer
        channels : int
            Number of channels
        format_type : int
            PyAudio format type
        """
        self.rate = rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format_type
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Buffer for storing recorded data
        self.frames = []
        
        # Flags for control
        self.is_recording = False
        self.stream = None
    
    def start_recording(self, max_seconds=None):
        """
        Start recording audio from microphone
        
        Parameters:
        ----------
        max_seconds : float or None
            Maximum recording time in seconds, if None, records until stop_recording is called
        """
        if self.is_recording:
            print("Already recording")
            return False
        
        try:
            self.frames = []
            self.stream = self.audio.open(
                format=self.format, 
                channels=self.channels,
                rate=self.rate, 
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.is_recording = True
            
            # If max_seconds is specified, schedule stop_recording
            if max_seconds is not None:
                def stop_after_time():
                    time.sleep(max_seconds)
                    if self.is_recording:
                        self.stop_recording()
                
                threading.Thread(target=stop_after_time, daemon=True).start()
            
            # Start recording thread
            threading.Thread(target=self._record_thread, daemon=True).start()
            
            print(f"Recording started{f' (max {max_seconds}s)' if max_seconds else ''}")
            return True
        except Exception as e:
            print(f"Error starting recording: {e}")
            return False
    
    def _record_thread(self):
        """Record audio in a separate thread"""
        while self.is_recording:
            try:
                data = self.stream.read(self.chunk_size)
                self.frames.append(data)
            except Exception as e:
                print(f"Error recording: {e}")
                self.is_recording = False
                break
    
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
        
        print("Recording stopped")
        return True
    
    def save_recording(self, file_path):
        """
        Save recorded audio to file
        
        Parameters:
        ----------
        file_path : str
            Path to save the audio file
            
        Returns:
        -------
        bool
            True if successful, False otherwise
        """
        if not self.frames:
            print("No recorded data to save")
            return False
        
        try:
            # Convert frames to numpy array
            audio_data = np.frombuffer(b''.join(self.frames), dtype=np.float32)
            
            # Save using soundfile
            sf.write(file_path, audio_data, self.rate)
            print(f"Recording saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving recording: {e}")
            return False
    
    def get_audio_data(self):
        """
        Get recorded audio data as numpy array
        
        Returns:
        -------
        np.ndarray
            Audio data
        int
            Sample rate
        """
        if not self.frames:
            print("No recorded data")
            return None, self.rate
        
        try:
            # Convert frames to numpy array
            audio_data = np.frombuffer(b''.join(self.frames), dtype=np.float32)
            return audio_data, self.rate
        except Exception as e:
            print(f"Error getting audio data: {e}")
            return None, self.rate
    
    def __del__(self):
        """Cleanup resources"""
        self.stop_recording()
        if self.audio:
            self.audio.terminate()
