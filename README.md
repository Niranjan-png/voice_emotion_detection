# Voice Emotion Detection

A web application that detects emotions from voice recordings in real-time.

## Features

- Real-time voice recording through the browser
- Emotion detection from audio input
- Visual representation of detected emotions with probabilities
- Interactive soundwave visualization during recording
- Support for both web interface and command-line real-time detection

## Demo

![Voice Emotion Detection Demo](https://example.com/demo.gif)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/voice-emotion-detection.git
   cd voice-emotion-detection
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Web Interface

1. Start the application:
   ```
   python run.py
   ```

2. Open a web browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```
   (or port 5000 if waitress is not installed)

3. Click "Start Recording" to begin recording audio from your microphone.

4. Speak for a few seconds, then click "Stop Recording".

5. The application will process your voice and display the detected emotion along with probability scores.

### Real-time Inference (Command Line)

You can also run the real-time inference script for continuous monitoring:

```
python -m voice_emotion_detection.realtime_inference
```

This will continuously capture audio from your microphone, analyze it, and display emotion predictions in real-time with visualizations.

## How It Works

The application uses a Convolutional Neural Network (CNN) model to detect emotions from audio signals. The process involves:

1. **Audio Capture**: Recording audio through the browser's MediaRecorder API or directly using PyAudio
2. **Audio Processing**: Converting the audio to an appropriate format for analysis
3. **Feature Extraction**: Extracting relevant features from the audio signal
4. **Emotion Prediction**: Using the CNN model to predict the emotion
5. **Visualization**: Displaying the results with probabilities

## Supported Emotions

- Angry
- Happy
- Neutral
- Sad
- Fearful
- Disgust
- Surprised

## Deployment Options

### Local Deployment

For production use, install the waitress WSGI server:
```
pip install waitress
```

Then run the application with:
```
python run.py
```

### Cloud Deployment

#### Heroku
1. Create a Heroku account and install Heroku CLI
2. Create an app: `heroku create your-app-name`
3. Deploy: `git push heroku main`

#### Docker
A Dockerfile is included for containerized deployment:
```
docker build -t voice-emotion-detection .
docker run -p 8000:8000 voice-emotion-detection
```

## Project Structure

```
voice-emotion-detection/
├── README.md
├── requirements.txt
├── run.py
├── voice_emotion_detection/
│   ├── __init__.py
│   ├── app.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── emotion_model.py
│   │   └── pretrained_model.py
│   ├── audio_processor.py
│   ├── feature_extractor.py
│   ├── realtime_inference.py
│   └── templates/
│       └── index.html
└── temp_audio/
    └── .gitkeep
```

## License

[MIT License](LICENSE)

## Contributors

- Your Name

## Acknowledgements

- The emotion detection model architecture is based on research in speech emotion recognition
- Web interface built with Flask and modern JavaScript 
