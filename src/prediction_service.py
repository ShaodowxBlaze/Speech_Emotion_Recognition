# src/prediction_service.py
import torch
import librosa
import numpy as np
from model import EnhancedEmotionClassifier

class EmotionPredictor:
    def __init__(self, model_path='models/best_model.pth'):
        # Initialize basic attributes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = 16000
        
        # Define emotion categories
        self.emotion_categories = [
            'angry', 'happy', 'sad', 'neutral',
            'frustrated', 'excited', 'surprised', 'fearful'
        ]
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.input_size = checkpoint.get('input_size', 672)  # Use get() with default value
        
        # Initialize model
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            # Initialize model with correct input size
            model = EnhancedEmotionClassifier(
                input_size=self.input_size,
                num_emotions=len(self.emotion_categories)
            ).to(self.device)
            
            # Load trained weights
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def extract_features(self, audio_path):
        """Extract features from audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Calculate statistics for MFCCs
            mfcc_stats = np.concatenate([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                np.min(mfccs, axis=1),
                np.max(mfccs, axis=1)
            ])
            
            # Calculate statistics for mel spectrogram
            mel_stats = np.concatenate([
                np.mean(mel_spec_db, axis=1),
                np.std(mel_spec_db, axis=1),
                np.min(mel_spec_db, axis=1),
                np.max(mel_spec_db, axis=1)
            ])
            
            # Combine and normalize features
            features = np.concatenate([mfcc_stats, mel_stats])
            features = (features - np.mean(features)) / (np.std(features) + 1e-6)
            
            # Pad or truncate to match input size
            if len(features) < self.input_size:
                features = np.pad(features, (0, self.input_size - len(features)))
            else:
                features = features[:self.input_size]
                
            return features
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            raise
    
    @torch.no_grad()
    def predict(self, audio_path):
        """Predict emotions from audio file"""
        try:
            # Extract features
            features = self.extract_features(audio_path)
            features = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Get predictions
            emotions, vad, _ = self.model(features)
            
            # Process emotion probabilities
            emotions = emotions.cpu().numpy()[0]
            emotions_prob = np.exp(emotions) / np.sum(np.exp(emotions))  # Apply softmax
            
            # Process VAD values
            vad = vad.cpu().numpy()[0]
            vad = np.tanh(vad)  # Normalize to [-1, 1]
            vad = (vad + 1) / 2  # Convert to [0, 1]
            
            # Filter and sort emotions
            emotion_dict = {
                emotion: float(prob)
                for emotion, prob in zip(self.emotion_categories, emotions_prob)
                if prob > 0.1  # Only include significant emotions
            }
            
            # Sort emotions by probability
            emotion_dict = dict(sorted(
                emotion_dict.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            # Create results dictionary
            predictions = {
                'emotions': emotion_dict,
                'dimensions': {
                    'valence': float(vad[0]),
                    'arousal': float(vad[1]),
                    'dominance': float(vad[2])
                },
                'interpretation': {
                    'valence': self._interpret_dimension(vad[0], 'valence'),
                    'arousal': self._interpret_dimension(vad[1], 'arousal'),
                    'dominance': self._interpret_dimension(vad[2], 'dominance')
                }
            }
            
            return predictions
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise
    
    def _interpret_dimension(self, value, dimension):
        """Interpret VAD values"""
        if dimension == 'valence':
            if value < 0.4:
                return "Negative"
            elif value < 0.6:
                return "Neutral"
            else:
                return "Positive"
        elif dimension == 'arousal':
            if value < 0.4:
                return "Calm"
            elif value < 0.6:
                return "Moderate"
            else:
                return "Excited"
        else:  # dominance
            if value < 0.4:
                return "Submissive"
            elif value < 0.6:
                return "Neutral"
            else:
                return "Dominant"