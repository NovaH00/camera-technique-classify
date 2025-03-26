from model import VideoClassify
import torch
import torch.nn as nn
import json
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VideoClassify(5, 150).to(device)

# Load the full state dictionary, not just model weights
checkpoint_path = r"150_frames\checkpoints\best_model.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)


# Full checkpoint with additional data
model.load_state_dict(checkpoint['model_state_dict'])
# Extract class_map from state
if 'class_map' in checkpoint:
    class_map = checkpoint['class_map']
    print("Loaded class_map from checkpoint:", class_map)
