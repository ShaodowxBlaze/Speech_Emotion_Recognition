import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from data_processing import EmotionDataProcessor

def build_dataset(iemocap_dir):
    """Process all IEMOCAP sessions and build dataset"""
    processor = EmotionDataProcessor()
    all_utterances = []
    
    # Process each session
    for session_id in range(1, 6):  # Sessions 1-5
        session = f"Ses{session_id:02d}"
        
        # Get all evaluation files for this session
        eval_dir = Path(iemocap_dir) / "Evaluation"
        audio_dir = Path(iemocap_dir) / "Audio"
        
        # Find all evaluation files
        eval_files = list(eval_dir.glob(f"{session}*.txt"))
        
        print(f"Processing Session {session}")
        for eval_file in tqdm(eval_files):
            # Get corresponding audio file
            audio_file = audio_dir / f"{eval_file.stem}.wav"
            
            if audio_file.exists():
                try:
                    # Process the file pair
                    utterances = processor.process_file_pair(str(audio_file), str(eval_file))
                    all_utterances.extend(utterances)
                    
                except Exception as e:
                    print(f"Error processing {eval_file.stem}: {str(e)}")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([{
        'utterance_id': u['utterance_id'],
        'emotion': u['emotion'],
        'categorical_emotions': u['categorical_emotions'],
        'valence': u['valence'],
        'arousal': u['arousal'],
        'dominance': u['dominance'],
        'mfcc': u['features']['mfcc'],
        'mel_spectrogram': u['features']['mel_spectrogram'],
        'audio': u['audio']
    } for u in all_utterances])
    
    # Save processed data
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save features and metadata
    df.to_pickle(output_dir / "processed_dataset.pkl")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total utterances: {len(df)}")
    print("\nEmotion distribution:")
    print(df['emotion'].value_counts())
    
    return df

if __name__ == "__main__":
    # Update this path to your IEMOCAP directory
    IEMOCAP_DIR = r"C:\Users\user\Desktop\FYP\Dataset\IEMOCAP_full_release"
    
    # Add tqdm to requirements.txt
    df = build_dataset(IEMOCAP_DIR)