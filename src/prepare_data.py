import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class EmotionDataset(Dataset):
    def __init__(self, features, emotions, vad_values):
        self.features = features
        self.emotions = emotions
        self.vad_values = vad_values
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.features[idx]),
            'emotions': torch.FloatTensor(self.emotions[idx]),
            'vad': torch.FloatTensor(self.vad_values[idx])
        }

def prepare_data(data_path="data/processed/processed_dataset.pkl", test_size=0.2, val_size=0.2):
    # Load the processed data
    print("Loading dataset...")
    df = pd.read_pickle(data_path)
    
    # Remove 'xxx' labeled data or use their categorical emotions
    df['main_emotion'] = df['emotion']
    xxx_mask = df['emotion'] == 'xxx'
    
    # For 'xxx' labeled data, use most frequent categorical emotion if available
    for idx in df[xxx_mask].index:
        if len(df.loc[idx, 'categorical_emotions']) > 0:
            # Get most common emotion from categorical_emotions
            emotion_counts = pd.Series(df.loc[idx, 'categorical_emotions']).value_counts()
            if len(emotion_counts) > 0:
                df.loc[idx, 'main_emotion'] = emotion_counts.index[0].lower()[:3]
    
    # Remove remaining xxx entries
    df = df[df['main_emotion'] != 'xxx']
    
    # Prepare features
    print("Preparing features...")
    
    # Combine MFCC and mel spectrogram features
    features = []
    for idx in df.index:
        mfcc = df.loc[idx, 'mfcc']
        mel_spec = df.loc[idx, 'mel_spectrogram']
        
        # Calculate statistics of these features
        mfcc_stats = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.min(mfcc, axis=1),
            np.max(mfcc, axis=1)
        ])
        
        mel_stats = np.concatenate([
            np.mean(mel_spec, axis=1),
            np.std(mel_spec, axis=1),
            np.min(mel_spec, axis=1),
            np.max(mel_spec, axis=1)
        ])
        
        features.append(np.concatenate([mfcc_stats, mel_stats]))
    
    features = np.array(features)
    
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Prepare emotion labels (multi-label format)
    emotion_categories = ['ang', 'hap', 'sad', 'neu', 'fru', 'exc', 'sur', 'fea']
    emotion_labels = np.zeros((len(df), len(emotion_categories)))
    
    for i, row in enumerate(df.itertuples()):
        main_emotion = row.main_emotion
        if main_emotion in emotion_categories:
            emotion_labels[i, emotion_categories.index(main_emotion)] = 1
            
        # Add categorical emotions as well
        for emotion in row.categorical_emotions:
            emotion_code = emotion.lower()[:3]
            if emotion_code in emotion_categories:
                emotion_labels[i, emotion_categories.index(emotion_code)] = 1
    
    # Prepare VAD values
    vad_values = df[['valence', 'arousal', 'dominance']].values
    
    # Split into train, validation, and test sets
    X_temp, X_test, y_temp, y_test, vad_temp, vad_test = train_test_split(
        features, emotion_labels, vad_values, test_size=test_size, random_state=42
    )
    
    X_train, X_val, y_train, y_val, vad_train, vad_val = train_test_split(
        X_temp, y_temp, vad_temp, test_size=val_size, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Save the scaler for later use
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(scaler, output_dir / "feature_scaler.joblib")
    
    # Create datasets
    train_dataset = EmotionDataset(X_train, y_train, vad_train)
    val_dataset = EmotionDataset(X_val, y_val, vad_val)
    test_dataset = EmotionDataset(X_test, y_test, vad_test)
    
    return train_dataset, val_dataset, test_dataset, emotion_categories

if __name__ == "__main__":
    # Add scikit-learn and joblib to requirements.txt
    train_dataset, val_dataset, test_dataset, emotion_cats = prepare_data()
    print("\nEmotion categories:", emotion_cats)