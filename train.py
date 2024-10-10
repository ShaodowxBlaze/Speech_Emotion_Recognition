import os
import numpy as np
import librosa
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.utils import to_categorical

print("Script is starting...")

def extract_mfcc_features(file_path):
    print(f"Extracting features from: {file_path}")
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)

        max_len = 1024
        if mfccs.shape[1] < max_len:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_len - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_len]

        return np.expand_dims(mfccs, axis=-1)
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

def extract_features(emotion_file, audio_dir):
    print("Starting feature extraction...")
    features = []
    labels = []
    skip_header_lines = 4

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    audio_file_set = set(audio_files)

    print(f"Found {len(audio_file_set)} audio files")

    try:
        with open(emotion_file, 'r') as f:
            for i, line in enumerate(f):
                if i < skip_header_lines or not line.strip():
                    continue

                parts = line.split(';')
                if len(parts) < 2:
                    print(f"Skipping line {i+1} due to insufficient parts: {line.strip()}")
                    continue
                
                # Get the audio filename part
                emotion_data = parts[0].strip()
                emotion_info = emotion_data.split()
                if len(emotion_info) < 2:
                    print(f"Skipping line {i+1} due to insufficient emotion data: {emotion_data}")
                    continue

                # Extract the emotion and corresponding audio file name
                emotion = emotion_info[1] if len(emotion_info) > 1 else None
                audio_file_base = emotion_info[0].replace(':', '').replace(' ', '')

                # Adjust to match the audio file naming convention
                audio_file_name = f"Ses01F_impro01.wav"  # Hardcoded for testing, update to match actual naming convention
                audio_file_path = os.path.join(audio_dir, audio_file_name)

                if not os.path.isfile(audio_file_path):
                    print(f"Audio file does not exist for turn '{audio_file_base}': {audio_file_path}")
                    continue

                mfcc_features = extract_mfcc_features(audio_file_path)
                if mfcc_features is not None:
                    features.append(mfcc_features)
                    labels.append(emotion)

        print(f"Number of features extracted: {len(features)}")
        print(f"Number of labels extracted: {len(labels)}")
        return features, labels
    except Exception as e:
        print(f"Error in extract_features: {str(e)}")
        return [], []

def create_model(input_shape, num_classes):
    print(f"Creating model with input shape {input_shape} and {num_classes} classes")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

if __name__ == '__main__':
    print("Entering main execution block...")
    try:
        session = 'Session1'
        audio_dir = os.path.join('C:/Users/user/Desktop/FYP/Dataset/IEMOCAP_full_release/Audio')
        emotion_file = os.path.join('C:/Users/user/Desktop/FYP/Dataset/IEMOCAP_full_release/Evaluation/Ses01F_impro01.txt')

        print(f"Audio directory: {audio_dir}")
        print(f"Emotion file: {emotion_file}")
        print(f"Checking if audio directory exists: {os.path.exists(audio_dir)}")
        print(f"Checking if emotion file exists: {os.path.exists(emotion_file)}")

        features, labels = extract_features(emotion_file, audio_dir)

        if not labels:
            print("No valid labels were extracted. Check the emotion file format.")
        else:
            print("Processing extracted features and labels...")
            emotion_labels = np.unique(labels)
            label_to_index = {label: index for index, label in enumerate(emotion_labels)}
            indexed_labels = [label_to_index[label] for label in labels]

            features = np.array(features)
            indexed_labels = np.array(indexed_labels)

            num_classes = len(emotion_labels)
            indexed_labels = to_categorical(indexed_labels, num_classes=num_classes)

            print(f"Features shape: {features.shape}")
            print(f"Labels shape: {indexed_labels.shape}")

            input_shape = features.shape[1:]
            model = create_model(input_shape, num_classes)

            print("Compiling model...")
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            print("Starting model training...")
            model.fit(features, indexed_labels, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

            print("Saving model...")
            model.save('C:/Users/user/Desktop/FYP/emotion_recognition_model.h5')
            print("Model saved successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

print("Script execution completed.")
