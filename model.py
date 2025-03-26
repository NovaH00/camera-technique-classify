from modules.architechtures import TransformerEncoderBlock
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

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
        
        # Smaller transformer to save memory
        self.transformer_block = TransformerEncoderBlock(input_dim=128, num_heads=4, ff_hidden_dim=128, dropout=0.1)
        
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
        
        # Apply transformer for temporal modeling
        temporal_features = self.transformer_block(spatial_features)  # (B, T, 128)
        
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