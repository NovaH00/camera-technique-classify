import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import VideoClassify
from dataset import create_data_loaders
from benchmark import count_parameters

# Training configuration
CONFIG = {
    'data_dir': './data',              # Path to dataset directory
    'output_dir': './checkpoints',     # Path to save checkpoints and logs
    'frames': 150,                      # Number of frames to sample per video
    'batch_size': 8,                   # Training batch size
    'val_batch_size': 16,              # Validation batch size
    'epochs': 50,                      # Number of training epochs
    'lr': 0.001,                       # Initial learning rate
    'weight_decay': 1e-4,              # Weight decay
    'num_workers': 4,                  # Number of data loading workers
    'save_freq': 5                     # Save checkpoint every N epochs
}

def train_model():
    """Main training function."""
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        root_dir=CONFIG['data_dir'],
        batch_size=CONFIG['batch_size'],
        num_frames=CONFIG['frames'],
        num_workers=CONFIG['num_workers']
    )
    
    # Get number of classes from dataset
    num_classes = len(train_loader.dataset.class_map)
    print(f"Training on {num_classes} classes")
    
    # Create model
    model = VideoClassify(num_classes=num_classes, frames=CONFIG['frames'])
    model = model.to(device)
    
    # Print model information
    count_parameters(model)
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training state
    start_epoch = 0
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(start_epoch, CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        best_val_acc = max(val_acc, best_val_acc)
        
        if is_best or (epoch + 1) % CONFIG['save_freq'] == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
            }, is_best)
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")

def train_epoch(model, data_loader, criterion, optimizer, device, epoch):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0
    
    start_time = time.time()
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    for inputs, labels in pbar:
        # Move inputs and labels to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()
        processed_size += inputs.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / processed_size,
            'acc': running_corrects / processed_size,
            'time': f"{time.time() - start_time:.2f}s"
        })
    
    # Calculate metrics
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects / len(data_loader.dataset)
    
    return epoch_loss, epoch_acc

def validate(model, data_loader, criterion, device):
    """Evaluate the model on the validation set."""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Validating"):
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
    
    # Calculate metrics
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects / len(data_loader.dataset)
    
    return epoch_loss, epoch_acc

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """Save training checkpoint."""
    checkpoint_path = os.path.join(CONFIG['output_dir'], filename)
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    if is_best:
        best_path = os.path.join(CONFIG['output_dir'], 'best_model.pth')
        torch.save(state, best_path)
        print(f"Best model saved to {best_path}")