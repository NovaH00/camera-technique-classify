import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path

# Add this at module level to track logged videos
_logged_video_rotations = set()

class VideoDataset(Dataset):
    """Dataset for loading video data for camera technique classification."""
    
    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        num_frames: int = 64,
        transform=None,
        temporal_sample_method: str = "uniform",
        frame_size: Tuple[int, int] = (224, 224),
    ):
        """
        Args:
            video_paths: List of paths to video files
            labels: List of class labels for each video
            num_frames: Number of frames to extract from each video
            transform: Optional transforms to apply to frames
            temporal_sample_method: Method to sample frames ("uniform" or "random")
            frame_size: Size to resize frames to (height, width)
        """
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform
        self.temporal_sample_method = temporal_sample_method
        self.frame_size = frame_size
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        return len(self.video_paths)
    
    def _load_video(self, video_path: str) -> np.ndarray:
        """Load video and extract frames with proper orientation."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            raise ValueError(f"Failed to load video: {video_path}")
        
        # Get rotation metadata
        rotation = 0
        try:
            # Try to get rotation metadata (OpenCV 4.5.1+)
            rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        except:
            # Fallback for older OpenCV versions
            try:
                stream_info = cv2.VideoCapture.get(cap, cv2.CAP_PROP_ORIENTATION_META)
                if stream_info:
                    rotation = int(stream_info)
            except:
                pass
        
        # Only log rotation information once per video file
        video_filename = os.path.basename(video_path)
        log_key = f"{video_filename}_{rotation}"
        if log_key not in _logged_video_rotations:
            print(f"Video {video_filename} rotation metadata: {rotation} degrees")
            _logged_video_rotations.add(log_key)
        
        # Choose frame indices based on sampling method
        if self.temporal_sample_method == "uniform":
            # Uniformly sample frames across the video
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        elif self.temporal_sample_method == "random":
            # Randomly sample frames
            indices = sorted(random.sample(range(total_frames), min(self.num_frames, total_frames)))
            # If we need more frames than the video has, we'll cycle through
            if len(indices) < self.num_frames:
                extra = np.random.choice(indices, self.num_frames - len(indices))
                indices = np.concatenate([indices, extra])
        else:
            raise ValueError(f"Unsupported temporal sampling method: {self.temporal_sample_method}")
        
        # Extract selected frames
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                # If frame reading fails, create a black frame
                frame = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
            else:
                # Apply rotation based on metadata - each frame is rotated only once
                if rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Convert from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize frame
                frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
            
            frames.append(frame)
        
        cap.release()
        return np.array(frames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get video frames and label for a given index."""
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video frames
        frames = self._load_video(video_path)
        
        # Apply transforms to each frame
        transformed_frames = []
        for frame in frames:
            if self.transform:
                frame = self.transform(frame)
            transformed_frames.append(frame)
        
        # Stack frames along time dimension
        # Output shape: (T, C, H, W)
        video_tensor = torch.stack(transformed_frames)
        
        # Rearrange to expected model input shape: (T, C, H, W)
        return video_tensor, label


def create_dataloader(
    data_dir: str,
    batch_size: int = 8,
    num_frames: int = 64,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    frame_size: Tuple[int, int] = (224, 224),
    video_extensions: Union[List[str], str] = "*.mp4",
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """
    Create train and validation dataloaders from a directory of videos.
    
    Args:
        data_dir: Directory containing class folders with videos
        batch_size: Batch size for dataloaders
        num_frames: Number of frames to sample from each video
        num_workers: Number of worker processes for dataloaders
        train_ratio: Ratio of data to use for training
        frame_size: Size to resize frames to (height, width)
        video_extensions: File extensions to look for (e.g., "*.mp4" or ["*.mp4", "*.avi"])
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        class_to_idx: Dictionary mapping class indices to class names
    """
    data_dir = Path(data_dir)
    
    # Check if data directory exists
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Get class folders
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {data_dir}. Please make sure your data is organized in subdirectories, with each subdirectory representing a class.")
    
    print(f"Found {len(class_dirs)} class directories: {[d.name for d in class_dirs]}")
    
    class_to_idx = {cls.name: i for i, cls in enumerate(class_dirs)}
    
    video_paths = []
    labels = []
    
    # Handle both string and list of extensions
    if isinstance(video_extensions, str):
        video_extensions = [video_extensions]
    
    # Collect video paths and labels
    for class_dir in class_dirs:
        class_idx = class_to_idx[class_dir.name]
        class_videos = []
        
        # Try multiple extensions
        for ext in video_extensions:
            class_videos.extend(list(class_dir.glob(ext)))
        
        if not class_videos:
            print(f"Warning: No videos found in {class_dir} with extensions {video_extensions}")
            continue
            
        print(f"Found {len(class_videos)} videos in class '{class_dir.name}'")
        
        for video_file in class_videos:
            video_paths.append(str(video_file))
            labels.append(class_idx)
    
    if len(video_paths) == 0:
        raise ValueError(f"No video files found in the class directories. "
                         f"Checked extensions: {video_extensions}. "
                         f"Please make sure your videos have the correct extensions.")
    
    print(f"Total videos found: {len(video_paths)}")
    
    # Create train/val split
    indices = list(range(len(video_paths)))
    random.shuffle(indices)
    split = int(train_ratio * len(indices))
    
    if split == 0:
        split = 1  # Ensure at least one sample in training set
    
    train_indices = indices[:split]
    val_indices = indices[split:] if split < len(indices) else [indices[0]]  # Ensure at least one sample in validation
    
    print(f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}")
    
    # Create transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(frame_size[0]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(frame_size[0] + 32),
        transforms.CenterCrop(frame_size[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = VideoDataset(
        [video_paths[i] for i in train_indices],
        [labels[i] for i in train_indices],
        num_frames=num_frames,
        transform=train_transform,
        temporal_sample_method="random",
        frame_size=frame_size
    )
    
    val_dataset = VideoDataset(
        [video_paths[i] for i in val_indices],
        [labels[i] for i in val_indices],
        num_frames=num_frames,
        transform=val_transform,
        temporal_sample_method="uniform",
        frame_size=frame_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),  # Ensure batch size isn't larger than dataset
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True if len(train_dataset) > batch_size else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(batch_size, len(val_dataset)),  # Ensure batch size isn't larger than dataset
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Invert class_to_idx for easier interpretation
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    return train_loader, val_loader, idx_to_class
