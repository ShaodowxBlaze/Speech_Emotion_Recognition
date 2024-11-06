# src/evaluate.py
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from model import EnhancedEmotionClassifier
import torch.nn.functional as F

class ModelEvaluator:
    def __init__(self, model_path, test_dataset, emotion_categories):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotion_categories = emotion_categories
        self.test_dataset = test_dataset
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model
        input_size = test_dataset[0]['features'].shape[0]
        self.model = EnhancedEmotionClassifier(
            input_size=input_size,
            num_emotions=len(emotion_categories)
        ).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def evaluate(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            pin_memory=True
        )
        
        all_predictions = []
        all_labels = []
        all_vad_pred = []
        all_vad_true = []
        
        print("\nEvaluating model on test set...")
        with torch.no_grad():
            for batch in tqdm(test_loader):
                features = batch['features'].to(self.device)
                emotion_labels = batch['emotions'].to(self.device)
                vad_labels = batch['vad'].to(self.device)
                
                # Get predictions
                emotions_logits, vad_pred, _ = self.model(features)
                emotions_prob = torch.sigmoid(emotions_logits)
                
                # Store predictions and labels
                all_predictions.extend(emotions_prob.cpu().numpy())
                all_labels.extend(emotion_labels.cpu().numpy())
                all_vad_pred.extend(vad_pred.cpu().numpy())
                all_vad_true.extend(vad_labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_vad_pred = np.array(all_vad_pred)
        all_vad_true = np.array(all_vad_true)
        
        # Calculate and display metrics
        self.calculate_emotion_metrics(all_predictions, all_labels)
        self.calculate_vad_metrics(all_vad_pred, all_vad_true)
        
        # Plot confusion matrices
        self.plot_confusion_matrices(all_predictions, all_labels)
        
        # Plot emotion distribution
        self.plot_emotion_distribution(all_predictions)
        
        # Save detailed results
        self.save_detailed_results(all_predictions, all_labels, all_vad_pred, all_vad_true)
    
    def calculate_emotion_metrics(self, predictions, labels, threshold=0.5):
        print("\nEmotion Recognition Metrics:")
        print("-" * 50)
        
        # Convert probabilities to binary predictions
        pred_binary = (predictions > threshold).astype(int)
        
        # Calculate metrics for each emotion
        for i, emotion in enumerate(self.emotion_categories):
            true_pos = np.sum((pred_binary[:, i] == 1) & (labels[:, i] == 1))
            false_pos = np.sum((pred_binary[:, i] == 1) & (labels[:, i] == 0))
            false_neg = np.sum((pred_binary[:, i] == 0) & (labels[:, i] == 1))
            true_neg = np.sum((pred_binary[:, i] == 0) & (labels[:, i] == 0))
            
            precision = true_pos / (true_pos + false_pos + 1e-10)
            recall = true_pos / (true_pos + false_neg + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            accuracy = (true_pos + true_neg) / len(labels)
            
            print(f"\n{emotion}:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
    
    def calculate_vad_metrics(self, predictions, labels):
        print("\nVAD Prediction Metrics:")
        print("-" * 50)
        
        dimensions = ['Valence', 'Arousal', 'Dominance']
        mae = np.mean(np.abs(predictions - labels), axis=0)
        mse = np.mean((predictions - labels)**2, axis=0)
        rmse = np.sqrt(mse)
        
        for i, dim in enumerate(dimensions):
            print(f"\n{dim}:")
            print(f"Mean Absolute Error: {mae[i]:.4f}")
            print(f"Root Mean Square Error: {rmse[i]:.4f}")
            print(f"Mean Square Error: {mse[i]:.4f}")
    
    def plot_confusion_matrices(self, predictions, labels, threshold=0.5):
        pred_binary = (predictions > threshold).astype(int)
        
        # Create directory for plots
        plots_dir = Path('plots')
        plots_dir.mkdir(exist_ok=True)
        
        plt.figure(figsize=(20, 15))
        for i, emotion in enumerate(self.emotion_categories):
            plt.subplot(3, 3, i+1)
            cm = confusion_matrix(labels[:, i], pred_binary[:, i])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {emotion}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'confusion_matrices.png')
        plt.close()
    
    def plot_emotion_distribution(self, predictions, threshold=0.5):
        pred_binary = (predictions > threshold).astype(int)
        emotion_counts = pred_binary.sum(axis=0)
        
        plt.figure(figsize=(12, 6))
        plt.bar(self.emotion_categories, emotion_counts)
        plt.title('Distribution of Predicted Emotions')
        plt.xticks(rotation=45)
        plt.ylabel('Count')
        
        plots_dir = Path('plots')
        plt.savefig(plots_dir / 'emotion_distribution.png', bbox_inches='tight')
        plt.close()
    
    def save_detailed_results(self, predictions, labels, vad_pred, vad_true):
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Save raw predictions and labels
        np.save(results_dir / 'predictions.npy', predictions)
        np.save(results_dir / 'true_labels.npy', labels)
        np.save(results_dir / 'vad_predictions.npy', vad_pred)
        np.save(results_dir / 'vad_true.npy', vad_true)

def main():
    from prepare_data import prepare_data
    
    try:
        # Load test dataset
        _, _, test_dataset, emotion_categories = prepare_data()
        
        # Initialize evaluator
        evaluator = ModelEvaluator(
            model_path='models/best_model.pth',
            test_dataset=test_dataset,
            emotion_categories=emotion_categories
        )
        
        # Run evaluation
        evaluator.evaluate()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()