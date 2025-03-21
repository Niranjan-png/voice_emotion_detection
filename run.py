from voice_emotion_detection.app import app

if __name__ == "__main__":
    try:
        from waitress import serve
        print("Starting Voice Emotion Detection Server with Waitress (Production)...")
        print("Go to http://127.0.0.1:8000 in your web browser")
        serve(app, host="0.0.0.0", port=8000)
    except ImportError:
        print("Waitress not installed. Using Flask's development server instead.")
        print("Note: This is not recommended for production use.")
        print("Go to http://127.0.0.1:5000 in your web browser")
        app.run(debug=True, host="0.0.0.0", port=5000)
