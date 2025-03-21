# Real-time inference module for emotion detection 

import numpy as np
import torch
import threading
import queue
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from audio_processor import AudioProcessor
from feature_extractor import FeatureExtractor
from models.pretrained_model import get_pretrained_model, get_emotion_label

class RealtimeEmotionDetector:
    """
    Real-time emotion detection from audio input
    """
    def __init__(self, 
                 sample_rate=16000,
                 window_size=2.0,
                 hop_size=0.5):
        """
        Initialize the real-time emotion detector
        
        Parameters:
        ----------
        sample_rate : int
            Audio sample rate
        window_size : float
            Size of the analysis window in seconds
        hop_size : float
            Hop size between windows in seconds
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        
        # Initialize components
        self.feature_extractor = FeatureExtractor(
            sample_rate=sample_rate,
            duration=window_size
        )
        
        # Initialize pre-trained model
        self.model = get_pretrained_model()
        
        self.audio_processor = AudioProcessor(
            rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            feature_extractor=self.feature_extractor
        )
        
        # Set callback for feature extraction
        self.audio_processor.set_on_features_callback(self._on_features)
        
        # Store current emotion and probabilities
        self.current_emotion = None
        self.current_probs = None
        self.emotion_history = []
        self.max_history = int(10 / hop_size)  # Store 10 seconds of history
        
        # Visualization
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.bar_plot = None
        self.line_plot = None
        self.is_visualizing = False
        self.viz_thread = None
        
    def _on_features(self, features):
        """Callback for processing extracted features"""
        try:
            # Convert features to appropriate format if needed
            # The model expects audio waveform, so we'll keep the raw audio
            audio_data = features  # Assuming features is the raw audio data
            
            # Make prediction using the pre-trained model
            result = self.model.generate(audio_data, extract_embedding=False)
            
            # Get emotion label and probabilities
            if 'labels' in result and 'scores' in result:
                label_idx = result['labels'][0]  # Get the predicted emotion index
                self.current_emotion = get_emotion_label(label_idx)
                self.current_probs = result['scores'][0]  # Get the probability scores
                
                # Update history
                self.emotion_history.append(self.current_probs)
                if len(self.emotion_history) > self.max_history:
                    self.emotion_history.pop(0)
                
        except Exception as e:
            print(f"Error in emotion detection: {e}")
    
    def _init_visualization(self):
        """Initialize visualization plots"""
        plt.ion()  # Enable interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Bar plot for current probabilities
        x = np.arange(9)  # 9 emotion classes
        self.bar_plot = self.ax1.bar(x, np.zeros(9))
        self.ax1.set_xticks(x)
        self.ax1.set_xticklabels([get_emotion_label(i) for i in range(9)])
        self.ax1.set_ylim(0, 1)
        self.ax1.set_title('Current Emotion Probabilities')
        
        # Line plot for emotion history
        self.line_plot = []
        for i in range(9):
            line, = self.ax2.plot([], [], label=get_emotion_label(i))
            self.line_plot.append(line)
        self.ax2.set_ylim(0, 1)
        self.ax2.set_title('Emotion Probability History')
        self.ax2.legend()
        
        plt.tight_layout()
    
    def _update_visualization(self):
        """Update visualization plots"""
        if self.current_probs is not None:
            # Update bar plot
            for bar, prob in zip(self.bar_plot, self.current_probs):
                bar.set_height(prob)
            
            # Update line plot
            if self.emotion_history:
                history_array = np.array(self.emotion_history)
                x = np.arange(len(history_array))
                for i, line in enumerate(self.line_plot):
                    line.set_data(x, history_array[:, i])
                self.ax2.set_xlim(0, len(history_array))
            
            plt.draw()
            plt.pause(0.01)
    
    def _visualization_loop(self):
        """Main visualization loop"""
        self._init_visualization()
        
        while self.is_visualizing:
            self._update_visualization()
            time.sleep(0.1)
        
        plt.close()
    
    def start(self):
        """Start real-time emotion detection"""
        # Start audio processing
        if not self.audio_processor.start_recording():
            return False
        
        # Start visualization
        self.is_visualizing = True
        self.viz_thread = threading.Thread(target=self._visualization_loop)
        self.viz_thread.daemon = True
        self.viz_thread.start()
        
        return True
    
    def stop(self):
        """Stop real-time emotion detection"""
        # Stop visualization
        self.is_visualizing = False
        if self.viz_thread:
            self.viz_thread.join(timeout=1.0)
        
        # Stop audio processing
        return self.audio_processor.stop_recording()
    
    def get_current_emotion(self):
        """
        Get current detected emotion and probabilities
        
        Returns:
        -------
        str
            Current emotion
        np.ndarray
            Emotion probabilities
        """
        return self.current_emotion, self.current_probs


def main():
    """Main function for running real-time emotion detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time emotion detection from voice')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate')
    parser.add_argument('--window_size', type=float, default=2.0, help='Analysis window size in seconds')
    parser.add_argument('--hop_size', type=float, default=0.5, help='Hop size between windows in seconds')
    
    args = parser.parse_args()
    
    # Create and start detector
    detector = RealtimeEmotionDetector(
        sample_rate=args.sample_rate,
        window_size=args.window_size,
        hop_size=args.hop_size
    )
    
    print("Starting emotion detection... Press Ctrl+C to stop")
    try:
        detector.start()
        while True:
            emotion, probs = detector.get_current_emotion()
            if emotion is not None:
                print(f"\rCurrent emotion: {emotion}", end='')
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping emotion detection...")
    finally:
        detector.stop()


if __name__ == '__main__':
    main() 
