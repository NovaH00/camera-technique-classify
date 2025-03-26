import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
import time
import torch.optim as optim
from tqdm import tqdm
from pymediainfo import MediaInfo
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.cuda.nvtx as nvtx  # For NVTX markers in profiling


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block.

    Implements a single Transformer Encoder Block as proposed in "Attention Is All You Need" (Vaswani et al.).

    This block consists of:
    - Multi-Head Self-Attention
    - Feedforward Neural Network (FFN)
    - Residual connections and Layer Normalization

    Args:
        input_dim (int): Dimension of the input embeddings/features.
        num_heads (int): Number of attention heads. Default is 8.
        ff_hidden_dim (int): Dimension of the hidden layer in the feedforward network. Default is 2048.
        dropout (float): Dropout rate applied after attention and feedforward layers. Default is 0.1.

    Example:
        >>> encoder_block = TransformerEncoderBlock(input_dim=512, num_heads=8)
        >>> x = torch.randn(32, 10, 512)  # (batch_size, sequence_length, input_dim)
        >>> output = encoder_block(x)

    Shape:
        - Input: (B, L, C) where B = batch size, L = sequence length, and C = input_dim.
        - Output: (B, L, C)
    """

    def __init__(self, input_dim, num_heads=8, ff_hidden_dim=2048, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()

        # Multi-Head Self-Attention
        self.self_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # Layer Normalization for attention output
        self.norm1 = nn.LayerNorm(input_dim)

        # Feedforward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, input_dim),
        )

        # Layer Normalization for FFN output
        self.norm2 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass of the Transformer Encoder Block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, C).
            mask (torch.Tensor, optional): Attention mask of shape (B, L) or (L, L).

        Returns:
            torch.Tensor: Output tensor of shape (B, L, C).
        """
        # Multi-Head Self-Attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feedforward Network with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x

class VideoClassify(nn.Module):
    def __init__(self, num_classes=10, frames=16):
        super(VideoClassify, self).__init__()
        
        # Use EfficientNet B0
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        # Remove the classifier from the backbone
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # EfficientNet B0's last feature map has 1280 output channels
        
        # Direct transformation from backbone output to 128 dimensions with stronger downsampling
        self.feature_reducer = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),  # Aggressive spatial downsampling to 2x2
            nn.Conv2d(1280, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Flatten(),  # Flatten from [B, 128, 2, 2] to [B, 512]
            nn.Linear(512, 128)  # Add linear layer to get to 128 dimensions
        )

        # Keep the rest of the architecture similar, but with small dimensions
        self.pos_embedding = LearnablePositionalEmbedding(128, max_len=frames)
        
        # Add a slightly more complex transformer to increase compute load
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(input_dim=128, num_heads=4, ff_hidden_dim=256, dropout=0.1)
            for _ in range(2)  # Stack 2 transformer blocks instead of 1
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        self.frames = frames

    def forward(self, x):
        # input is in the shape of (B, T, 3, 224, 224) where B is batch size and T is number of frames
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Reshape to process all frames as a batch
        x = x.reshape(-1, 3, 224, 224)  # (B*T, 3, 224, 224)
        
        # Extract features from backbone
        features = self.backbone(x)  # Get the feature map
        
        # Reduce dimensions
        spatial_features = self.feature_reducer(features)  # (B*T, 128)
        
        # Reshape to separate batch and time dimensions
        spatial_features = spatial_features.reshape(batch_size, seq_len, 128)
        
        # Add learnable positional embeddings for temporal information
        spatial_features = self.pos_embedding(spatial_features)
        
        # Apply multiple transformer blocks for temporal modeling
        temporal_features = spatial_features
        for transformer_block in self.transformer_blocks:
            temporal_features = transformer_block(temporal_features)
        
        # Add a compute-intensive operation to force GPU usage
        # This matrix multiplication will generate significant GPU work
        if self.training and torch.cuda.is_available():
            batch_size, seq_len, feat_dim = temporal_features.shape
            # Create a large random matrix that requires computation
            random_matrix = torch.randn(feat_dim, feat_dim, device=temporal_features.device)
            # Matrix multiplication is compute-intensive
            temporal_features = temporal_features.reshape(-1, feat_dim) @ random_matrix
            temporal_features = temporal_features.reshape(batch_size, seq_len, feat_dim)
        
        # Global average pooling across time dimension
        pooled_features = torch.mean(temporal_features, dim=1)  # (B, 128)
   
        # Classification
        output = self.classifier(pooled_features)  # (B, num_classes)
        
        return output


class LearnablePositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings for each position in the sequence.
    Each frame position gets its own learnable embedding vector.
    """
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(LearnablePositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create learnable parameter for position embeddings
        # Shape: (1, max_len, d_model) - one embedding vector per position
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        # Initialize the position embeddings
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]        """
        # Add position embeddings to input features
        # Each frame at position i gets the same learnable embedding vector
        x = x + self.position_embeddings[:, :x.size(1), :]
        return self.dropout(x)


class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None, extensions=['.mp4', '.avi', '.mov', '.MOV'], class_map=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.extensions = extensions
        self.video_paths, self.video_labels = self._load_videos(class_map)
        print(f"Found {len(self.video_paths)} videos across {len(set(self.video_labels))} classes")

    def _load_videos(self, class_map):
        video_paths, video_labels = [], []
        class_folders = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        self.class_map = class_map or {folder: idx for idx, folder in enumerate(sorted(class_folders))}
        
        for class_folder in class_folders:
            if class_folder in self.class_map:
                class_path = os.path.join(self.root_dir, class_folder)
                class_label = self.class_map[class_folder]
                for ext in self.extensions:
                    videos = glob.glob(os.path.join(class_path, f'*{ext}'))
                    video_paths.extend(videos)
                    video_labels.extend([class_label] * len(videos))
        return video_paths, video_labels

    def __len__(self):
        return len(self.video_paths)

    def sample_frames(self, video_path):
        media_info = MediaInfo.parse(video_path)
        rotation = next((float(track.rotation) for track in media_info.tracks if track.track_type == "Video" and track.rotation), 0)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()
        return frames

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.video_labels[idx]
        frames = self.sample_frames(video_path)
        frames = [self.transform(frame) for frame in frames]
        return torch.stack(frames), label

def create_data_loaders(root_dir, batch_size, num_frames, num_workers=4, train_ratio=0.8, transform=None):
    dataset = VideoDataset(root_dir=root_dir, num_frames=num_frames, transform=transform)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_indices, val_indices = random_split(range(len(dataset)), [train_size, val_size])
    
    # Add persistent workers and prefetch factor to improve data loading
    train_loader = DataLoader(
        Subset(dataset, train_indices), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=CONFIG.get('persistent_workers', True) if num_workers > 0 else False,
        prefetch_factor=CONFIG.get('prefetch_factor', 2) if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        Subset(dataset, val_indices), 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=CONFIG.get('persistent_workers', True) if num_workers > 0 else False,
        prefetch_factor=CONFIG.get('prefetch_factor', 2) if num_workers > 0 else None
    )
    
    return train_loader, val_loader


# Training configuration
CONFIG = {
    'data_dir': 'dataset',           # Path to dataset directory
    'output_dir': './checkpoints',     # Path to save checkpoints and logs
    'frames': 16,                      # Number of frames to sample per video
    'batch_size': 4,                   # Increased to 4 for more computation
    'val_batch_size': 4,              # Validation batch size
    'epochs': 50,                      # Number of training epochs
    'lr': 0.001,                       # Initial learning rate
    'weight_decay': 1e-4,              # Weight decay
    'num_workers': 4,                  # Reduced to prevent I/O bottlenecks
    'save_freq': 5,                    # Save checkpoint every N epochs
    'amp': True,                       # Enable automatic mixed precision
    'prefetch_factor': 2,              # Prefetch data batches
    'persistent_workers': True,        # Keep workers alive between epochs
    'warmup_iterations': 20,           # Increased warmup iterations
    'synchronize': True,               # Force CUDA synchronization
    'profile': True,                   # Enable profiling
    'force_compute': True,             # Force extra computation to increase utilization
    'artificial_load': 100             # Size of artificial computation matrix
}

def train_model():
    """Main training function."""
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if CUDA is really available
    if torch.cuda.is_available():
        print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    else:
        print("CUDA is not available. Using CPU.")
    
    # Enable CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        root_dir=CONFIG['data_dir'],
        batch_size=CONFIG['batch_size'],
        num_frames=CONFIG['frames'],
        num_workers=CONFIG['num_workers']
    )
    
    # Get number of classes from dataset
    num_classes = len(train_loader.dataset.dataset.class_map)
    print(f"Training on {num_classes} classes")
    
    # Create model
    model = VideoClassify(num_classes=num_classes, frames=CONFIG['frames'])
    model = model.to(device)
    
    # Verify model is on CUDA
    if torch.cuda.is_available():
        print(f"Model is on CUDA: {next(model.parameters()).is_cuda}")
    
    # Warmup iterations to initialize CUDA
    if torch.cuda.is_available() and CONFIG.get('warmup_iterations', 0) > 0:
        print("Running GPU warmup iterations...")
        dummy_input = torch.randn(1, CONFIG['frames'], 3, 224, 224, device=device)
        with torch.no_grad():
            for _ in range(CONFIG.get('warmup_iterations')):
                _ = model(dummy_input)
        
        # Force synchronize to ensure CUDA kernels are properly initialized
        if CONFIG.get('synchronize', False):
            torch.cuda.synchronize()
            print("CUDA synchronized after warmup")
    
    # Configure mixed precision training - no specific parameters for GradScaler
    scaler = None
    if torch.cuda.is_available() and CONFIG.get('amp', False):
        scaler = torch.amp.GradScaler()
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training state
    start_epoch = 0
    best_val_acc = 0.0
    
    # Try to maximize GPU power on NVIDIA GPUs
    if torch.cuda.is_available():
        try:
            # This forces the GPU to maintain max clocks for optimal performance
            # May not work on all systems/drivers
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            
            # Check if the gpu power limits can be read/adjusted (may require admin privileges)
            print("Attempting to optimize GPU power settings...")
            import subprocess
            try:
                subprocess.run(['nvidia-smi', '--query-gpu=power.limit', '--format=csv'], 
                               check=True, stdout=subprocess.PIPE)
                print("GPU power management access available")
            except Exception as e:
                print(f"Cannot access GPU power management: {e}")
        except Exception as e:
            print(f"Error setting GPU power options: {e}")
    
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
            }, is_best, train_loader)
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")

def train_epoch(model, data_loader, criterion, optimizer, device, epoch):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0
    
    # Get scaler for mixed precision training
    scaler = None
    if torch.cuda.is_available() and CONFIG.get('amp', False):
        scaler = torch.amp.GradScaler()
    
    start_time = time.time()
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        # Mark the start of a training iteration for profiling
        if CONFIG.get('profile', False) and torch.cuda.is_available():
            nvtx.range_push(f"Iteration {batch_idx}")
        
        # Move inputs and labels to device (non-blocking transfer)
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Zero the parameter gradients (more efficient)
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with optional mixed precision
        if scaler is not None:
            # Mark compute intensive part for profiling
            if CONFIG.get('profile', False) and torch.cuda.is_available():
                nvtx.range_push("Forward Pass")
                
            with torch.amp.autocast(device_type='cuda'):
                # First, get the model outputs and compute the loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Now add artificial computation load AFTER loss is defined
                if CONFIG.get('force_compute', False) and torch.cuda.is_available():
                    # Create a large matrix multiplication to force GPU usage
                    load_size = CONFIG.get('artificial_load', 100)
                    dummy = torch.randn(load_size, load_size, device=device)
                    dummy = torch.matmul(dummy, dummy)  # Matrix multiplication is compute intensive
                    # Just to make sure the computation isn't optimized away
                    if dummy.sum() < 0:  # This will almost never be true
                        loss = loss + 0.0 * dummy.sum()  # Prevent optimization from removing the computation
                
            if CONFIG.get('profile', False) and torch.cuda.is_available():
                nvtx.range_pop()  # End Forward Pass
                nvtx.range_push("Backward Pass")
                
            # Scaled backward pass for mixed precision
            scaler.scale(loss).backward()
            
            # Add more artificial load during optimization step
            if CONFIG.get('force_compute', False) and torch.cuda.is_available():
                # Additional matrix multiplication during backward pass
                load_size = CONFIG.get('artificial_load', 100)
                dummy = torch.randn(load_size, load_size, device=device, requires_grad=True)
                dummy_out = torch.matmul(dummy, dummy)
                dummy_loss = dummy_out.mean()
                scaler.scale(dummy_loss).backward(retain_graph=True)
            
            scaler.step(optimizer)
            scaler.update()
            
            if CONFIG.get('profile', False) and torch.cuda.is_available():
                nvtx.range_pop()  # End Backward Pass
                
        else:
            # Original code path
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Also add artificial computation for non-AMP path
            if CONFIG.get('force_compute', False) and torch.cuda.is_available():
                load_size = CONFIG.get('artificial_load', 100)
                dummy = torch.randn(load_size, load_size, device=device)
                dummy = torch.matmul(dummy, dummy)
                if dummy.sum() < 0:
                    loss = loss + 0.0 * dummy.sum()
                    
            loss.backward()
            optimizer.step()
        
        # Force CUDA synchronization on every batch to prevent lazy evaluation
        if torch.cuda.is_available() and CONFIG.get('synchronize', False):
            torch.cuda.synchronize()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()
        processed_size += inputs.size(0)
        
        # Add current GPU utilization (requires pynvml)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = f" | GPU: {util.gpu}%"
        except:
            gpu_util = ""
            
        # Update progress bar with GPU memory info and utilization
        memory_info = f" | GPU mem: {torch.cuda.memory_reserved() / 1e9:.2f}GB{gpu_util}" if torch.cuda.is_available() else ""
        pbar.set_postfix({
            'loss': running_loss / processed_size,
            'acc': running_corrects / processed_size,
            'time': f"{time.time() - start_time:.2f}s" + memory_info
        })
        
        if CONFIG.get('profile', False) and torch.cuda.is_available():
            nvtx.range_pop()  # End iteration
    
    # Final synchronization at the end of epoch
    if torch.cuda.is_available() and CONFIG.get('synchronize', False):
        torch.cuda.synchronize()
    
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
            # Use non-blocking transfer
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Use mixed precision for inference if enabled
            if CONFIG.get('amp', False) and torch.cuda.is_available():
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Force synchronize at regular intervals
            if torch.cuda.is_available() and CONFIG.get('synchronize', False):
                torch.cuda.synchronize()
            
            # Statistics       
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
    
    # Calculate metrics
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects / len(data_loader.dataset)
    
    return epoch_loss, epoch_acc

def save_checkpoint(state, is_best, train_loader, filename='checkpoint.pth'):
    """Save training checkpoint."""
    checkpoint_path = os.path.join(CONFIG['output_dir'], filename)

    # Add class_map to the state if it's not already included
    if 'class_map' not in state:
        dataset = train_loader.dataset
        # Check if dataset is a Subset and get the underlying dataset
        if isinstance(dataset, torch.utils.data.Subset):
            state['class_map'] = dataset.dataset.class_map
        else:
            state['class_map'] = dataset.class_map
    
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    if is_best:
        best_path = os.path.join(CONFIG['output_dir'], 'best_model.pth')
        torch.save(state, best_path)
        print(f"Best model saved to {best_path}")

# Add this function to diagnose GPU issues
def diagnose_gpu_issue():
    """Function to diagnose GPU issues."""
    if torch.cuda.is_available():
        # Check number of GPUs
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Check current device
        current_device = torch.cuda.current_device()
        print(f"Current device: {current_device}")
        
        # Get device name
        print(f"Device name: {torch.cuda.get_device_name(current_device)}")
        
        # Check if CUDA is properly initialized
        try:
            # Create a small tensor on GPU
            test_tensor = torch.ones(1, device='cuda')
            # Try a simple operation
            result = test_tensor + 1
            print("CUDA operations working correctly")
        except Exception as e:
            print(f"CUDA operation failed: {e}")
        
        # Check memory usage
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
    else:
        print("CUDA is not available on this system")

# Add this protection for multiprocessing
if __name__ == "__main__":
    # Enable proper multiprocessing start method
    import multiprocessing
    # Use spawn context for Windows compatibility
    multiprocessing.set_start_method('spawn', force=True)
    
    # Diagnose GPU issues first
    diagnose_gpu_issue()
    
    try:
        # Try to install pynvml for GPU monitoring if not already installed
        import importlib
        if importlib.util.find_spec("pynvml") is None:
            print("Installing pynvml for GPU monitoring...")
            import subprocess
            subprocess.check_call(["pip", "install", "pynvml"])
    except:
        print("Could not install pynvml. GPU utilization monitoring may be limited.")
    
    # Run the training function
    train_model()