import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import os

def download_emotion_model(model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", output_dir="pretrained"):
    """
    Download the pre-trained emotion recognition model
    
    Parameters:
    ----------
    model_name : str
        Model name from HuggingFace
    output_dir : str
        Directory to save the model
    """
    print(f"Downloading model {model_name}...")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Download model and feature extractor
        model = AutoModelForAudioClassification.from_pretrained(model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        # Save model and feature extractor
        model.save_pretrained(os.path.join(output_dir, "model"))
        feature_extractor.save_pretrained(os.path.join(output_dir, "feature_extractor"))
        
        print(f"Model downloaded successfully to {output_dir}")
        return model, feature_extractor
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None, None

def test_model(model):
    """Test the model with a sample audio file"""
    if model is None:
        print("No model available for testing")
        return
    
    try:
        # Get example audio file path
        example_path = os.path.join(model.model_path, "example", "test.wav")
        if not os.path.exists(example_path):
            print("No example file found for testing")
            return
            
        # Run inference
        result = model.generate(
            example_path,
            output_dir="./outputs",
            granularity="utterance",
            extract_embedding=False
        )
        
        print("\nTest Results:")
        print("-------------")
        if 'labels' in result:
            emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised', 'unknown']
            scores = result.get('scores', [])
            for emotion, score in zip(emotions, scores[0]):
                print(f"{emotion}: {score:.4f}")
        else:
            print("Features extracted:", result['feats'].shape)
            
    except Exception as e:
        print(f"Error testing model: {e}")

if __name__ == "__main__":
    # Download the model
    model, feature_extractor = download_emotion_model()
    if model is not None:
        print("\nModel architecture:")
        print(model)
        test_model(model) 
