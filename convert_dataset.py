import os
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from typing import List, Tuple, Dict, Optional, Union
import json


# Set to track videos that have already been logged for rotation
_logged_video_rotations = set()

def get_video_rotation(video_path: str) -> int:
    """
    Extract rotation metadata from a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        int: Rotation angle in degrees (0, 90, 180, or 270)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
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
    
    cap.release()
    return rotation

def extract_frames(video_path: str, num_frames: int = 64, frame_size: Tuple[int, int] = (224, 224), 
                  temporal_sample_method: str = "uniform") -> np.ndarray:
    """
    Extract frames from a video file and apply proper rotation.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        frame_size: Size to resize frames to (height, width)
        temporal_sample_method: Method to sample frames ("uniform" or "random")
        
    Returns:
        np.ndarray: Array of extracted frames with shape (num_frames, height, width, 3)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    
    # Get rotation metadata
    rotation = get_video_rotation(video_path)
    
    # Choose frame indices based on sampling method
    if temporal_sample_method == "uniform":
        # Uniformly sample frames across the video
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    elif temporal_sample_method == "random":
        # Randomly sample frames
        indices = sorted(np.random.choice(range(total_frames), min(num_frames, total_frames), replace=False))
        # If we need more frames than the video has, we'll cycle through
        if len(indices) < num_frames:
            extra = np.random.choice(indices, num_frames - len(indices))
            indices = np.concatenate([indices, extra])
    else:
        raise ValueError(f"Unsupported temporal sampling method: {temporal_sample_method}")
    
    # Extract selected frames
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # If frame reading fails, create a black frame
            frame = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
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
            frame = cv2.resize(frame, (frame_size[1], frame_size[0]))
        
        frames.append(frame)
    
    cap.release()
    return np.array(frames)

def apply_transforms(frames: np.ndarray, transform_type: str = "train") -> torch.Tensor:
    """
    Apply transforms to a batch of frames.
    
    Args:
        frames: Array of frames with shape (num_frames, height, width, 3)
        transform_type: Type of transform to apply ("train" or "val")
        
    Returns:
        torch.Tensor: Tensor of transformed frames with shape (num_frames, 3, height, width)
    """
    # Define transforms
    if transform_type == "train":
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # val
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Apply transforms to each frame
    transformed_frames = []
    for frame in frames:
        transformed_frames.append(transform(frame))
    
    # Stack frames along time dimension
    return torch.stack(transformed_frames)

def process_dataset(input_dir: str, output_dir: str, num_frames: int = 64, train_ratio: float = 0.8,
                   video_extensions: List[str] = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.MOV"]) -> None:
    """
    Process a video dataset by extracting frames, applying rotation, and saving as tensor files.
    
    Args:
        input_dir: Input directory containing class subdirectories with videos
        output_dir: Output directory to save processed data
        num_frames: Number of frames to extract from each video
        train_ratio: Ratio of data to use for training
        video_extensions: File extensions to look for
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    output_train_dir = output_dir / "train"
    output_val_dir = output_dir / "val"
    
    # Check if input directory exists
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Get class folders
    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if len(class_dirs) == 0:
        raise ValueError(f"No class directories found in {input_dir}")
    
    print(f"Found {len(class_dirs)} class directories: {[d.name for d in class_dirs]}")
    
    # Process each class
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"Processing class: {class_name}")
        
        # Create output directories for this class
        train_class_dir = output_train_dir / class_name
        val_class_dir = output_val_dir / class_name
        
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all videos in this class
        class_videos = []
        for ext in video_extensions:
            class_videos.extend(list(class_dir.glob(ext)))
        
        if not class_videos:
            print(f"Warning: No videos found in {class_dir} with extensions {video_extensions}")
            continue
        
        print(f"Found {len(class_videos)} videos in class '{class_name}'")
        
        # Split videos into train and validation sets
        np.random.shuffle(class_videos)
        split_idx = int(len(class_videos) * train_ratio)
        train_videos = class_videos[:split_idx]
        val_videos = class_videos[split_idx:]
        
        # Process training videos
        for video_file in tqdm(train_videos, desc=f"Processing train videos for '{class_name}'"):
            try:
                # Extract frames and apply rotation
                frames = extract_frames(
                    str(video_file), 
                    num_frames=num_frames, 
                    frame_size=(224, 224), 
                    temporal_sample_method="random"
                )
                
                # Apply transforms for training
                tensor_frames = apply_transforms(frames, transform_type="train")
                
                # Save tensor to file
                output_file = train_class_dir / f"{video_file.stem}.pt"
                torch.save(tensor_frames, output_file)
            except Exception as e:
                print(f"Error processing video {video_file}: {e}")
        
        # Process validation videos
        for video_file in tqdm(val_videos, desc=f"Processing val videos for '{class_name}'"):
            try:
                # Extract frames and apply rotation
                frames = extract_frames(
                    str(video_file), 
                    num_frames=num_frames, 
                    frame_size=(224, 224), 
                    temporal_sample_method="uniform"
                )
                
                # Apply transforms for validation
                tensor_frames = apply_transforms(frames, transform_type="val")
                
                # Save tensor to file
                output_file = val_class_dir / f"{video_file.stem}.pt"
                torch.save(tensor_frames, output_file)
            except Exception as e:
                print(f"Error processing video {video_file}: {e}")
    
    print(f"Dataset conversion complete. Processed data saved to {output_dir}")
    
    # Save class mapping
    class_mapping = {i: name for i, name in enumerate([d.name for d in class_dirs])}
    with open(output_dir / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Convert video dataset to tensor files for faster training")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing class folders with videos")
    parser.add_argument("--output", type=str, required=True, help="Output directory to save processed data")
    parser.add_argument("--num-frames", type=int, default=64, help="Number of frames to extract from each video")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of data to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Process dataset
    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        num_frames=args.num_frames,
        train_ratio=args.train_ratio
    )

if __name__ == "__main__":
    main()
