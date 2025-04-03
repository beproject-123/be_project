from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model/deepvoice_detector.h5')
max_length = joblib.load('model/max_length.joblib')

def extract_features(audio_path):
    """Extract MFCC features from audio file"""
    try:
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        
        # Add delta and delta-delta features for better accuracy
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # Combine features
        combined_features = np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=0)
        
        # Transpose for model input
        return combined_features.T
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def process_audio(audio_path):
    """Process audio file and make prediction"""
    # Extract features
    features = extract_features(audio_path)
    if features is None:
        return {"error": "Failed to process audio file"}
    
    # Pad sequence
    padded_features = pad_sequences([features], maxlen=max_length, dtype='float32', 
                                   padding='post', truncating='post')
    
    # Make prediction
    prediction = model.predict(padded_features)
    result = prediction[0]
    
    # Return result
    label = "Fake" if result[1] > result[0] else "Real"
    confidence = float(max(result))
    
    return {
        "prediction": label,
        "confidence": confidence,
        "raw_scores": {
            "real": float(result[0]),
            "fake": float(result[1])
        }
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No audio file selected"}), 400
    
    # Save file temporarily
    temp_path = os.path.join('temp', audio_file.filename)
    os.makedirs('temp', exist_ok=True)
    audio_file.save(temp_path)
    
    # Process file
    result = process_audio(temp_path)
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return jsonify(result)

if __name__ == '__main__':
    os.makedirs('temp', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    app.run(debug=True)