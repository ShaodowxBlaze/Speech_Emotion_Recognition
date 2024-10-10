import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model (ensure the correct path to the model is specified)
model_path = os.path.join('C:/Users/user/Desktop/FYP/', 'emotion_recognition_model.h5')
model = load_model(model_path)

# Emotion labels based on the IEMOCAP dataset (modify according to your actual labels)
emotion_labels = ['happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised', 'neutral', 'excited', 'bored'] # Adjust as needed

# Define Flask app
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('C:/Users/user/Desktop/FYP', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav'}

# Function to check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract MFCC features from uploaded audio file
def extract_features(file_path):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    print("Audio data shape:", audio_data.shape)  # Debugging info
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    
    max_len = 1024  # Adjust based on the model's expected input size
    if mfccs.shape[1] < max_len:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_len - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :max_len]

    return np.expand_dims(mfccs, axis=-1)  # Add an extra dimension for Conv2D input

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            file_path = save_file(file)
            features = extract_features(file_path)
            features = np.expand_dims(features, axis=0)  # Add batch dimension
            print("Features shape:", features.shape)  # Debugging info

            # Prediction
            try:
                prediction = model.predict(features)
                print("Raw prediction:", prediction)  # Debugging info
                predicted_index = np.argmax(prediction, axis=1)[0]  # Get index of the highest predicted value
                print("Predicted index:", predicted_index)  # Debugging info

                # Check if predicted_index is within the range of emotion_labels
                if predicted_index < len(emotion_labels):
                    predicted_emotion = emotion_labels[predicted_index]  # Map index to emotion label
                    return jsonify({'predicted_emotion': predicted_emotion})
                else:
                    return jsonify({'error': 'Prediction index out of range'}), 500

            except Exception as prediction_error:
                print(f"Prediction error: {prediction_error}")  # Log the error
                return jsonify({'error': f"Error during prediction: {str(prediction_error)}"}), 500
        return jsonify({'error': 'File not allowed'}), 400
    except Exception as e:
        print(f"File processing error: {e}")  # Log the error
        return jsonify({'error': f"Error processing the file: {str(e)}"}), 500
# Function to save the uploaded file
def save_file(file):
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    return file_path

if __name__ == '__main__':
    app.run(debug=True)
