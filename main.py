import os
import pandas as pd
import numpy as np
import librosa

# Function to load evaluation data and audio files
def load_data(session):
    evaluation_path = f'C:/Users/user/Desktop/FYP/Dataset/IEMOCAP_full_release/{session}/dialog/EmoEvaluation'
    audio_path = f'C:/Users/user/Desktop/FYP/Dataset/IEMOCAP_full_release/{session}/dialog/wav'

    evaluations = []
    audio_data = {}

    # Check if the EmoEvaluation folder exists
    if not os.path.exists(evaluation_path):
        raise FileNotFoundError(f"The evaluation folder does not exist: {evaluation_path}")

    for eval_file in os.listdir(evaluation_path):
        if eval_file.endswith('.txt'):
            eval_file_path = os.path.join(evaluation_path, eval_file)
            with open(eval_file_path, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip the header line if present
                    parts = line.strip().split()  # Adjust based on the structure of the evaluation file
                    evaluations.append(parts)

    # Load audio files
    for audio_file in os.listdir(audio_path):
        if audio_file.endswith('.wav'):
            audio_file_path = os.path.join(audio_path, audio_file)
            # Load audio data
            try:
                signal, sr = librosa.load(audio_file_path, sr=None)
                audio_data[audio_file] = (signal, sr)
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")

    # Convert evaluations to DataFrame for easier processing
    if evaluations:
        evaluations_df = pd.DataFrame(evaluations)
    else:
        evaluations_df = pd.DataFrame()  # In case no evaluation files are found

    return evaluations_df, audio_data

# Main code execution
if __name__ == '__main__':
    session = 'Session1'  # Specify the session you want to analyze
    evaluations_df, audio_data = load_data(session)

    # Display the evaluations and audio data information
    if not evaluations_df.empty:
        print("Evaluations DataFrame:")
        print(evaluations_df.head())
    else:
        print("No evaluations found.")

    print("\nAudio Data:")
    if audio_data:
        for key, value in audio_data.items():
            print(f"{key}: Sample Rate = {value[1]}, Signal Length = {len(value[0])}")
    else:
        print("No audio data found.")
