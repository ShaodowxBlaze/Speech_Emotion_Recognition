# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from pathlib import Path
import numpy as np
from tqdm import tqdm
import gc
from torch.optim.lr_scheduler import OneCycleLR

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=1000):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class EmotionAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        weights = self.attention(x)
        weights = F.softmax(weights, dim=1)
        weighted_sum = (x * weights).sum(dim=1)
        return weighted_sum, weights

class DualPathAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.global_attention = nn.MultiheadAttention(input_dim, num_heads=8, dropout=0.1)
        self.local_attention = nn.MultiheadAttention(input_dim, num_heads=4, dropout=0.1)
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        # Global path
        global_out, _ = self.global_attention(x, x, x)
        
        # Local path with sliding window
        batch_size, seq_len, dim = x.size()
        window_size = 10
        local_out = torch.zeros_like(x)
        
        for i in range(0, seq_len, window_size // 2):
            end = min(i + window_size, seq_len)
            window = x[:, i:end, :]
            local_attn, _ = self.local_attention(window, window, window)
            local_out[:, i:end, :] += local_attn
        
        # Fusion
        combined = torch.cat([global_out, local_out], dim=-1)
        return self.fusion(combined)

class EnhancedEmotionClassifier(nn.Module):
    def __init__(self, input_size, num_emotions):
        super().__init__()
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512)
        )
        
        # Convolutional layers for temporal patterns
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )
        
        # Attention mechanisms
        self.dual_attention = DualPathAttention(512)
        self.emotion_attention = EmotionAttention(512)
        
        # Emotion-specific feature extraction
        self.emotion_specific = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(256)
            ) for _ in range(4)  # Different pathways for different emotion aspects
        ])
        
        # Output layers
        self.emotion_classifier = nn.Linear(256 * 4, num_emotions)
        self.vad_regressor = nn.Linear(256 * 4, 3)
        
        # Auxiliary outputs
        self.arousal_classifier = nn.Linear(256 * 4, 1)
        self.valence_classifier = nn.Linear(256 * 4, 1)
    
    def forward(self, x):
        # Initial feature extraction
        x = self.feature_extractor(x)
        
        # Reshape for conv1d
        x = x.unsqueeze(1)
        
        # Temporal pattern extraction
        x = self.conv_layers(x)
        
        # Reshape for attention
        x = x.transpose(1, 2)
        
        # Apply dual attention
        x = self.dual_attention(x)
        
        # Apply emotion-specific attention
        attended_features, attention_weights = self.emotion_attention(x)
        
        # Multiple emotion-specific pathways
        emotion_features = []
        for pathway in self.emotion_specific:
            features = pathway(attended_features)
            emotion_features.append(features)
        
        # Combine features
        combined_features = torch.cat(emotion_features, dim=1)
        
        # Main outputs
        emotions = self.emotion_classifier(combined_features)
        vad = self.vad_regressor(combined_features)
        
        # Auxiliary outputs
        arousal = self.arousal_classifier(combined_features)
        valence = self.valence_classifier(combined_features)
        
        return emotions, vad, (arousal, valence, attention_weights)
    
    # Continuing from previous part...

class EmotionTrainer:
    def __init__(self, train_dataset, val_dataset, emotion_categories):
        self.device = self._setup_gpu()
        self.batch_size = 64  # Increased batch size for stability
        
        # Calculate class weights for balanced loss
        self.class_weights = self._calculate_class_weights(train_dataset)
        
        # Data loaders
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
            persistent_workers=True
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=2,
            persistent_workers=True
        )
        
        # Model initialization
        input_size = train_dataset[0]['features'].shape[0]
        self.model = EnhancedEmotionClassifier(input_size, len(emotion_categories)).to(self.device)
        
        # Loss functions with class weights
        self.emotion_criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        self.vad_criterion = nn.MSELoss()
        self.auxiliary_criterion = nn.MSELoss()
        
        # Optimizer with lower learning rate and increased weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            amsgrad=True
        )
        
        # OneCycleLR scheduler for better training stability
        self.scheduler = OneCycleLR(
    self.optimizer,
    max_lr=1e-3,
    epochs=100,
    steps_per_epoch=len(self.train_loader),
    pct_start=0.1,
    div_factor=25,
    final_div_factor=1000
)
        
        # Gradient scaler for mixed precision training
        self.scaler = amp.GradScaler()
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
        
        # Mixup parameters
        self.use_mixup = True
        self.mixup_alpha = 0.2
        
        # Gradient clipping value
        self.max_grad_norm = 0.5
    
    def _setup_gpu(self):
        """Setup GPU and return device"""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        return device
    
    def _calculate_class_weights(self, dataset):
        """Calculate class weights for balanced loss"""
        all_labels = torch.stack([sample['emotions'] for sample in dataset])
        pos_counts = torch.sum(all_labels, dim=0)
        total = len(dataset)
        weights = total / (2 * pos_counts)
        return weights.to(self.device)
    
    def mixup_data(self, x, y):
        """Perform mixup augmentation on the input and target"""
        if np.random.random() > 0.5:
            return x, y
            
        batch_size = x.size(0)
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y
    
    def train(self, num_epochs=100):
        for epoch in range(num_epochs):
            torch.cuda.empty_cache()
            gc.collect()
            
            # Training phase
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch in progress_bar:
                features = batch['features'].to(self.device, non_blocking=True)
                emotion_labels = batch['emotions'].to(self.device, non_blocking=True)
                vad_labels = batch['vad'].to(self.device, non_blocking=True)
                
                # Apply mixup augmentation
                if self.use_mixup:
                    features, emotion_labels = self.mixup_data(features, emotion_labels)
                    features, vad_labels = self.mixup_data(features, vad_labels)
                
                with torch.cuda.amp.autocast():
                    emotions, vad, aux_outputs = self.model(features)
                    arousal, valence, attention_weights = aux_outputs
                    
                    # Calculate losses
                    emotion_loss = self.emotion_criterion(emotions, emotion_labels)
                    vad_loss = self.vad_criterion(vad, vad_labels)
                    aux_arousal_loss = self.auxiliary_criterion(arousal.squeeze(), vad_labels[:, 1])
                    aux_valence_loss = self.auxiliary_criterion(valence.squeeze(), vad_labels[:, 0])
                    
                    # Attention regularization
                    attention_reg = torch.norm(attention_weights, p=2)
                    
                    # Combined loss with adjusted weights
                    loss = (
                        emotion_loss + 
                        0.3 * vad_loss + 
                        0.1 * aux_arousal_loss +
                        0.1 * aux_valence_loss +
                        0.01 * attention_reg
                    )
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                self.scheduler.step()
                
                train_loss += loss.item()
                train_steps += 1
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'gpu_mem': f"{torch.cuda.memory_allocated()/1024**2:.0f}MB",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
                del features, emotion_labels, vad_labels
            
            # Validation phase
            val_loss = self.validate()
            
            # Print epoch results
            avg_train_loss = train_loss / train_steps
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**2:.0f}MB')
            
            # Save best model with exponential moving average
            if epoch == 0:
                self.best_val_loss = val_loss
            else:
                self.best_val_loss = 0.9 * self.best_val_loss + 0.1 * val_loss
                
            if val_loss < self.best_val_loss:
                self.patience_counter = 0
                self.save_model(epoch, val_loss)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print("Early stopping triggered!")
                    break
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_loss = 0
        val_steps = 0
        
        for batch in self.val_loader:
            features = batch['features'].to(self.device, non_blocking=True)
            emotion_labels = batch['emotions'].to(self.device, non_blocking=True)
            vad_labels = batch['vad'].to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                emotions, vad, aux_outputs = self.model(features)
                arousal, valence, attention_weights = aux_outputs
                
                emotion_loss = self.emotion_criterion(emotions, emotion_labels)
                vad_loss = self.vad_criterion(vad, vad_labels)
                aux_arousal_loss = self.auxiliary_criterion(arousal.squeeze(), vad_labels[:, 1])
                aux_valence_loss = self.auxiliary_criterion(valence.squeeze(), vad_labels[:, 0])
                attention_reg = torch.norm(attention_weights, p=2)
                
                loss = (
                    emotion_loss + 
                    0.3 * vad_loss + 
                    0.1 * aux_arousal_loss +
                    0.1 * aux_valence_loss +
                    0.01 * attention_reg
                )
            
            val_loss += loss.item()
            val_steps += 1
            
            del features, emotion_labels, vad_labels
        
        return val_loss / val_steps
    
    def save_model(self, epoch, val_loss):
        save_path = Path('models')
        save_path.mkdir(exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
        }, save_path / 'best_model.pth')

if __name__ == "__main__":
    from prepare_data import prepare_data
    
    try:
        # Get data
        train_dataset, val_dataset, test_dataset, emotion_categories = prepare_data()
        
        # Initialize and train
        trainer = EmotionTrainer(train_dataset, val_dataset, emotion_categories)
        trainer.train()
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\nGPU out of memory! Try reducing batch size.")
        raise e
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise e