# Real-Time Voice Emotion Detection System

This project implements a real-time emotion detection system that classifies emotions from voice inputs using deep learning.

## Features

- Audio data preprocessing (spectrograms, MFCCs, pitch, zero-crossing rate)
- Noise reduction and normalization
- Deep learning model for emotion classification (using PyTorch)
- Real-time inference with PyAudio for audio capture
- User interface for emotion visualization

## Supported Emotions

The system can detect the following emotions:
- Angry
- Disgusted
- Fearful
- Happy
- Neutral
- Sad
- Surprised

## Project Structure

```
voice_emotion_detection/
├── data/               # For storing datasets and processed features
├── models/             # Model definition and saved models
├── utils/              # Utility functions for audio processing
├── audio_processor.py  # Audio preprocessing pipeline
├── feature_extractor.py # Feature extraction from audio
├── model_trainer.py    # Model training script
├── realtime_inference.py # Real-time emotion detection
├── app.py              # Main application with UI
└── README.md           # This file
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   ```
   # On Windows
   venv\Scripts\activate
   ```
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

1. Place your dataset in the `data` folder
2. Run the training script:
   ```
   python model_trainer.py
   ```

### Real-time Emotion Detection

Run the application:
```
python app.py
```

## Datasets

The model can be trained on various emotion datasets such as:
- RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- CREMA-D (Crowd-Sourced Emotional Multimodal Actors Dataset)
- IEMOCAP (Interactive Emotional Dyadic Motion Capture Database)
- EmoDB (Berlin Database of Emotional Speech)

## Model Architecture

The default model architecture is a CNN-LSTM hybrid network that combines the spatial feature extraction capabilities of CNNs with the temporal modeling of LSTMs.

## License

MIT
