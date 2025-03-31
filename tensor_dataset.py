import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Dict, Optional, List

class TensorVideoDataset(Dataset):
    """Dataset for loading pre-processed video tensor data."""
    
    def __init__(self, data_dir: str, split: str = "train"):
        """
        Args:
            data_dir: Base directory containing 'train' and 'val' folders
            split: 'train' or 'val'
        """
        self.data_dir = Path(data_dir) / split
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Get class directories
        self.class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        if len(self.class_dirs) == 0:
            raise ValueError(f"No class directories found in {self.data_dir}")
        
        # Load class mapping if available
        class_mapping_file = Path(data_dir) / "class_mapping.json"
        if class_mapping_file.exists():
            with open(class_mapping_file, "r") as f:
                self.class_mapping = json.load(f)
            # Convert string keys to integers
            self.class_mapping = {int(k): v for k, v in self.class_mapping.items()}
        else:
            # Create class mapping from directory names
            self.class_mapping = {i: d.name for i, d in enumerate(self.class_dirs)}
        
        # Create reverse mapping (name -> index)
        self.name_to_idx = {name: idx for idx, name in self.class_mapping.items()}
        
        # Collect tensor files and labels
        self.tensor_files = []
        self.labels = []
        
        for class_dir in self.class_dirs:
            class_name = class_dir.name
            class_idx = self.name_to_idx[class_name]
            
            for tensor_file in class_dir.glob("*.pt"):
                self.tensor_files.append(tensor_file)
                self.labels.append(class_idx)
        
        print(f"Loaded {len(self.tensor_files)} {split} samples from {len(self.class_dirs)} classes")
    
    def __len__(self):
        return len(self.tensor_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get tensor and label for a given index."""
        tensor_path = self.tensor_files[idx]
        label = self.labels[idx]
        
        # Load tensor from file
        video_tensor = torch.load(tensor_path)
        
        return video_tensor, label

def create_tensor_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """
    Create train and validation dataloaders from pre-processed tensor files.
    
    Args:
        data_dir: Directory containing 'train' and 'val' folders with processed data
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for dataloaders
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        class_mapping: Dictionary mapping class indices to class names
    """
    # Create datasets
    train_dataset = TensorVideoDataset(data_dir, split="train")
    val_dataset = TensorVideoDataset(data_dir, split="val")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True if len(train_dataset) > batch_size else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(batch_size, len(val_dataset)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, train_dataset.class_mapping
