import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch import nn

class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emotions = ['angry', 'happy', 'neutral', 'sad', 'fearful', 'disgust', 'surprised']
        
        # Audio preprocessing layers
        self.mel_spec = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            win_length=None,
            hop_length=512,
            n_mels=80
        )
        self.amp_to_db = T.AmplitudeToDB()
        
        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 20 * 32, 512)
        self.fc2 = nn.Linear(512, len(self.emotions))
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Ensure input is float and normalized
        x = x.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.max() > 1.0:
            x = x / x.max()
            
        # Convert to mel spectrogram
        x = self.mel_spec(x)
        x = self.amp_to_db(x)
        
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # CNN layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def get_pretrained_model(model_dir="models/pretrained"):
    """
    Get or initialize the pretrained emotion recognition model
    """
    os.makedirs(model_dir, exist_ok=True)
    model = EmotionCNN()
    model.eval()
    return model

def get_emotion_label(index):
    """Map index to emotion label"""
    emotions = ['angry', 'happy', 'neutral', 'sad', 'fearful', 'disgust', 'surprised']
    return emotions[index] if 0 <= index < len(emotions) else 'unknown'

if __name__ == "__main__":
    # Test the model initialization
    model = get_pretrained_model()
    print("Model initialized successfully!") 
