import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from model import TimeSFormer
from dataset import create_dataloader

# Configuration dictionary
CONFIG = {
    # Data parameters
    "data_dir": "dataset",  # Directory containing video data
    "output_dir": "output",  # Output directory for logs and checkpoints
    "num_frames": 64,  # Number of frames to sample from each video
    "train_ratio": 0.8,  # Ratio of data to use for training
    "video_extensions": ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.MOV"],  # Video file extensions to look for
    
    # Training parameters
    "batch_size": 4,  # Training batch size
    "epochs": 30,  # Number of training epochs
    "learning_rate": 1e-4,  # Initial learning rate
    "min_lr": 1e-6,  # Minimum learning rate
    "weight_decay": 1e-4,  # Weight decay coefficient
    "save_every": 5,  # Save checkpoint every N epochs
    
    # Other parameters
    "seed": 42,  # Random seed
    "num_workers": 4,  # Number of data loading workers
    "no_cuda": False,  # Disable CUDA
}


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')
    
    for videos, labels in progress_bar:
        # Move data to device
        videos = videos.to(device)  # Expects shape (B, T, C, H, W)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (progress_bar.n + 1),
            'acc': 100. * correct / total
        })
    
    train_loss = running_loss / len(dataloader)
    train_acc = 100. * correct / total
    
    return train_loss, train_acc


def validate(model, dataloader, criterion, device):
    """Evaluate the model on the validation set."""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation')
        
        for videos, labels in progress_bar:
            # Move data to device
            videos = videos.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            # Save predictions and targets for metric computation
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)})
    
    # Compute metrics
    val_loss = running_loss / len(dataloader)
    val_acc = accuracy_score(all_targets, all_predictions) * 100
    precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    
    metrics = {
        'val_loss': val_loss,
        'val_acc': val_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_dir):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))


def main(config):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() and not config["no_cuda"] else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["seed"])
    
    # Create output directories
    output_dir = Path(config["output_dir"])
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    
    for directory in [output_dir, checkpoint_dir, log_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Check if data directory exists
    data_dir = Path(config["data_dir"])
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        print(f"Current working directory: {Path.cwd()}")
        print(f"Please create the directory or update the CONFIG['data_dir'] value.")
        return
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    try:
        # Create data loaders - add back the video_extensions parameter
        train_loader, val_loader, idx_to_class = create_dataloader(
            config["data_dir"],
            batch_size=config["batch_size"],
            num_frames=config["num_frames"],
            num_workers=config["num_workers"],
            train_ratio=config["train_ratio"],
            frame_size=(224, 224),
            video_extensions=config["video_extensions"]  # Pass the video extensions from CONFIG
        )
        
        # Save class mapping
        with open(output_dir / 'class_mapping.json', 'w') as f:
            json.dump(idx_to_class, f, indent=4)
        
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        print(f"Class mapping: {idx_to_class}")
        
        # Initialize model
        num_classes = len(idx_to_class)
        model = TimeSFormer(number_of_frames=config["num_frames"], num_classes=num_classes)
        model = model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        
        # Learning rate scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["min_lr"])
        
        # Track best model
        best_val_acc = 0.0
        
        # Training loop
        for epoch in range(config["epochs"]):
            # Train for one epoch
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            
            # Validate
            val_metrics = validate(model, val_loader, criterion, device)
            
            # Update learning rate
            scheduler.step()
            
            # Log metrics
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Loss/val', val_metrics['val_loss'], epoch)
            writer.add_scalar('Accuracy/val', val_metrics['val_acc'], epoch)
            writer.add_scalar('Precision/val', val_metrics['precision'], epoch)
            writer.add_scalar('Recall/val', val_metrics['recall'], epoch)
            writer.add_scalar('F1/val', val_metrics['f1'], epoch)
            writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)
            
            # Print metrics
            print(f"Epoch {epoch+1}/{config['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_acc']:.2f}%")
            print(f"  Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # Save checkpoint if it's the best model so far
            if val_metrics['val_acc'] > best_val_acc:
                best_val_acc = val_metrics['val_acc']
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics, 
                    os.path.join(checkpoint_dir, 'best_model')
                )
                print(f"  New best model saved with val acc: {best_val_acc:.2f}%")
            
            # Regularly save checkpoints
            if (epoch + 1) % config["save_every"] == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics, checkpoint_dir
                )
        
        # Save final model
        save_checkpoint(
            model, optimizer, scheduler, config["epochs"] - 1, val_metrics, checkpoint_dir
        )
        
        writer.close()
        print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        writer.close()


if __name__ == "__main__":
    # Run with configuration from the CONFIG dictionary
    main(CONFIG)
