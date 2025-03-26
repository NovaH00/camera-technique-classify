import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image

class VideoDataset(Dataset):
    """
    Dataset for loading videos from a directory structure.
    
    The dataset expects videos to be organized in folders where each folder represents
    a class. It samples frames at even intervals to match the required number of frames
    for the VideoClassify model.
    
    Args:
        root_dir (str): Root directory containing video folders.
        num_frames (int): Number of frames to sample from each video.
        transform (callable, optional): Optional transform to be applied on sampled frames.
        extensions (list): List of valid video file extensions. Default: ['.mp4', '.avi', '.mov']
        class_map (dict, optional): Dictionary mapping folder names to class indices.
    """
    def __init__(self, root_dir, num_frames=16, transform=None, 
                 extensions=['.mp4', '.avi', '.mov'], class_map=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.extensions = extensions
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Find all video files and their corresponding classes
        self.video_paths = []
        self.video_labels = []
        
        # Get class folders
        class_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        # Create class map if not provided
        if class_map is None:
            self.class_map = {folder: idx for idx, folder in enumerate(sorted(class_folders))}
        else:
            self.class_map = class_map
        
        # Collect videos and their labels
        for class_folder in class_folders:
            if class_folder in self.class_map:
                class_path = os.path.join(root_dir, class_folder)
                class_label = self.class_map[class_folder]
                
                # Find video files with specified extensions
                for ext in self.extensions:
                    videos = glob.glob(os.path.join(class_path, f'*{ext}'))
                    for video_path in videos:
                        self.video_paths.append(video_path)
                        self.video_labels.append(class_label)
        
        print(f"Found {len(self.video_paths)} videos across {len(class_folders)} classes")
    
    def __len__(self):
        return len(self.video_paths)
    
    def sample_frames(self, video_path):
        """
        Sample frames at even intervals from a video.
        
        Args:
            video_path (str): Path to the video file.
            
        Returns:
            list: List of sampled frames as PIL Images.
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"Video has no frames: {video_path}")
        
        # Calculate frame indices to sample
        if total_frames <= self.num_frames:
            # If video has fewer frames than needed, duplicate frames
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # Sample frames at even intervals
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                # If frame read failed, create a black frame
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Convert BGR (OpenCV) to RGB (PIL)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def __getitem__(self, idx):
        """
        Get video frames and label for a given index.
        
        Args:
            idx (int): Index of the video to get.
            
        Returns:
            tuple: (frames, label) where frames is a tensor of shape (num_frames, 3, H, W)
            and label is the class index.
        """
        video_path = self.video_paths[idx]
        label = self.video_labels[idx]
        
        # Sample frames from the video
        frames = self.sample_frames(video_path)
        
        # Apply transforms to each frame
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        # Stack frames to create a tensor of shape (T, C, H, W)
        frames_tensor = torch.stack(frames)
        
        return frames_tensor, label

# Example usage to create data loaders
def create_data_loaders(root_dir, batch_size=8, num_frames=16, num_workers=4, 
                       train_transform=None, val_transform=None, train_ratio=0.8):
    """
    Create train and validation data loaders for video classification.
    
    Args:
        root_dir (str): Root directory containing video folders
        batch_size (int): Batch size for data loaders
        num_frames (int): Number of frames to sample per video
        num_workers (int): Number of workers for data loading
        train_transform (callable, optional): Transform for training data
        val_transform (callable, optional): Transform for validation data
        train_ratio (float): Ratio of training data (0.0 to 1.0)
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader, random_split
    
    # Default transforms
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Create full dataset
    full_dataset = VideoDataset(root_dir=root_dir, num_frames=num_frames, transform=None)
    
    # Split dataset into train and validation
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create dataset objects with appropriate transforms
    train_dataset = VideoDataset(
        root_dir=root_dir,
        num_frames=num_frames,
        transform=train_transform,
        class_map=full_dataset.class_map,
    )
    
    val_dataset = VideoDataset(
        root_dir=root_dir,
        num_frames=num_frames,
        transform=val_transform,
        class_map=full_dataset.class_map,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
