# src/train_gpu.py
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from pathlib import Path
import gc
from tqdm import tqdm

def set_gpu_settings():
    """Optimize GPU settings for training"""
    # Enable CUDA memory allocation optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
    print(f"Using GPU: {gpu_name}")
    print(f"Total GPU Memory: {gpu_memory:.2f} GB")
    
    return torch.device('cuda')

class EmotionTrainer:
    def __init__(self, train_dataset, val_dataset, emotion_categories):
        self.device = set_gpu_settings()
        
        # RTX 3050 Laptop has around 4GB VRAM, so let's optimize batch size
        self.batch_size = 128  # We can adjust this if needed
        
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
        
        # Initialize model
        input_size = train_dataset[0]['features'].shape[0]
        self.model = EnhancedEmotionClassifier(input_size, len(emotion_categories)).to(self.device)
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            amsgrad=True
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.001,
            epochs=50,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3
        )
        
        # Gradient scaler for mixed precision training
        self.scaler = amp.GradScaler()
        
        # Track metrics
        self.best_val_loss = float('inf')
        self.patience = 7
        self.patience_counter = 0
        
    def train(self, num_epochs=50):
        for epoch in range(num_epochs):
            # Clear GPU cache at start of epoch
            torch.cuda.empty_cache()
            gc.collect()
            
            # Training phase
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch in progress_bar:
                # Move data to GPU
                features = batch['features'].to(self.device, non_blocking=True)
                emotion_labels = batch['emotions'].to(self.device, non_blocking=True)
                vad_labels = batch['vad'].to(self.device, non_blocking=True)
                
                # Mixed precision training
                with amp.autocast():
                    emotions_pred, vad_pred = self.model(features)
                    
                    # Calculate losses
                    emotion_loss = nn.BCELoss()(emotions_pred, emotion_labels)
                    vad_loss = nn.MSELoss()(vad_pred, vad_labels)
                    loss = emotion_loss + 0.5 * vad_loss
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update learning rate
                self.scheduler.step()
                
                train_loss += loss.item()
                train_steps += 1
                
                # Update progress bar with GPU stats
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'gpu_mem': f"{torch.cuda.memory_allocated()/1024**2:.0f}MB",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # Clear memory
                del features, emotion_labels, vad_labels
            
            avg_train_loss = train_loss / train_steps
            
            # Validation phase
            val_loss = self.validate()
            
            # Print epoch results
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**2:.0f}MB')
            
            # Save best model and early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
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
            
            with amp.autocast():
                emotions_pred, vad_pred = self.model(features)
                emotion_loss = nn.BCELoss()(emotions_pred, emotion_labels)
                vad_loss = nn.MSELoss()(vad_pred, vad_labels)
                loss = emotion_loss + 0.5 * vad_loss
            
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
        # Prepare data
        train_dataset, val_dataset, test_dataset, emotion_categories = prepare_data()
        
        # Initialize trainer
        trainer = EmotionTrainer(train_dataset, val_dataset, emotion_categories)
        
        # Start training
        trainer.train()
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\nGPU out of memory! Try reducing batch size.")
        raise e
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise e