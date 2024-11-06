import librosa
import numpy as np
import re
from pathlib import Path

class EmotionDataProcessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def load_audio(self, audio_path):
        """Load audio file and return audio data"""
        try:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            raise Exception(f"Error loading audio file: {str(e)}")
            
    def load_evaluation(self, eval_path):
        """Load emotion evaluation file and parse the data"""
        evaluations = []
        current_eval = None
        
        try:
            with open(eval_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # Match the utterance line pattern
                    utterance_match = re.match(r'\[(.+?) - (.+?)\]\s+(\S+)\s+(\S+)\s+\[(.+?)\]', line)
                    if utterance_match:
                        if current_eval:
                            evaluations.append(current_eval)
                            
                        start_time = float(utterance_match.group(1))
                        end_time = float(utterance_match.group(2))
                        utterance_id = utterance_match.group(3)
                        emotion = utterance_match.group(4)
                        vad = [float(x) for x in utterance_match.group(5).split(',')]
                        
                        current_eval = {
                            'start_time': start_time,
                            'end_time': end_time,
                            'utterance_id': utterance_id,
                            'emotion': emotion,
                            'valence': vad[0],
                            'arousal': vad[1],
                            'dominance': vad[2],
                            'categorical_emotions': []
                        }
                    
                    # Match category evaluator lines
                    elif line.startswith('C-'):
                        if current_eval:
                            emotions = re.findall(r'(\w+);', line)
                            current_eval['categorical_emotions'].extend(emotions)
            
            if current_eval:
                evaluations.append(current_eval)
                
            return evaluations
        
        except Exception as e:
            raise Exception(f"Error loading evaluation file: {str(e)}")
            
    def extract_features(self, audio, sr=None):
        """Extract audio features"""
        if sr is None:
            sr = self.sample_rate
            
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            return {
                'mfcc': mfccs,
                'mel_spectrogram': mel_spec_db
            }
        except Exception as e:
            raise Exception(f"Error extracting features: {str(e)}")
        
    def process_file_pair(self, audio_path, eval_path):
        """Process an audio file and its corresponding evaluation file"""
        try:
            # Load audio
            audio = self.load_audio(audio_path)
            
            # Load evaluations
            evaluations = self.load_evaluation(eval_path)
            
            # Process each utterance
            utterances = []
            for eval_data in evaluations:
                # Extract utterance audio
                start_sample = int(eval_data['start_time'] * self.sample_rate)
                end_sample = int(eval_data['end_time'] * self.sample_rate)
                utterance_audio = audio[start_sample:end_sample]
                
                # Extract features
                features = self.extract_features(utterance_audio)
                
                utterances.append({
                    'utterance_id': eval_data['utterance_id'],
                    'audio': utterance_audio,
                    'features': features,
                    'emotion': eval_data['emotion'],
                    'categorical_emotions': eval_data['categorical_emotions'],
                    'valence': eval_data['valence'],
                    'arousal': eval_data['arousal'],
                    'dominance': eval_data['dominance']
                })
                
            return utterances
            
        except Exception as e:
            raise Exception(f"Error processing files: {str(e)}")
