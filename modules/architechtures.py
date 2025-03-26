import torch
import torch.nn as nn
import torch.nn.functional as F



class ResNetBlock(nn.Module):
    """
    ResNet Block.

    This block is a residual learning unit from the ResNet architecture. 
    It consists of three convolutional layers, each followed by Batch Normalization 
    and ReLU activation, along with a skip connection to enable residual learning.

    Args:
        input_channel (int): Number of input channels.
        output_channel (int): Number of output channels. Default is 256.

    Example:
        >>> resnet_block = ResNetBlock(64, 256)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = resnet_block(x)

    Shape:
        - Input: (B, C_in, H, W)
        - Output: (B, C_out, H, W) [same H and W as input if stride = 1]
    """

    def __init__(self, input_channel, output_channel=256):
        """
        Initializes the ResNet Block.

        Args:
            input_channel (int): Number of input channels.
            output_channel (int): Number of output channels. Default is 256.
        """
        super(ResNetBlock, self).__init__()
        
        # First 1x1 Convolution to reduce dimensions
        self.conv_stack_1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # 3x3 Convolution to extract features
        self.conv_stack_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Final 1x1 Convolution to increase dimensions to output_channel
        self.conv_stack_3 = nn.Sequential(
            nn.Conv2d(64, output_channel, kernel_size=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
        )
        
        # Skip Connection (adjust dimensions if needed)
        self.skip_connection = nn.Sequential()
        if input_channel != output_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1),
                nn.BatchNorm2d(output_channel),
            )
        
    def forward(self, x):
        """
        Forward pass of the ResNet Block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H, W).
        """
        # Residual path
        logits = self.conv_stack_1(x)
        logits = self.conv_stack_2(logits)
        logits = self.conv_stack_3(logits)
        
        # Skip connection
        skip = self.skip_connection(x)

        # Add residual and apply ReLU activation
        return F.relu(logits + skip)

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block.

    This block is inspired by the ConvNeXt architecture, which simplifies 
    the ResNet-style design with modern Transformer-like components. 
    It includes depthwise convolutions, Layer Normalization, pointwise convolutions, 
    and residual connections to enhance feature extraction.

    Args:
        input_channel (int): Number of input channels.
        output_channel (int): Number of output channels. Default is 96.

    Example:
        >>> convnext_block = ConvNeXtBlock(64, 96)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = convnext_block(x)

    Shape:
        - Input: (B, C, H, W)
        - Output: (B, C_out, H, W)
    """

    def __init__(self, input_channel, output_channel=96):
        """
        Initializes the ConvNeXt Block.

        Args:
            input_channel (int): Number of input channels.
            output_channel (int): Number of output channels. Default is 96.
        """
        super(ConvNeXtBlock, self).__init__()
        
        # Depthwise Convolution
        self.d_conv_stack = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, kernel_size=7, padding=3, groups=input_channel),
        )      
        
        # Layer Normalization (applied along channel dimension)
        self.layer_norm = nn.LayerNorm(input_channel, eps=1e-6)
        
        # Pointwise Convolutions (1x1 Convolutions for feature transformation)
        self.conv_stack_1 = nn.Sequential(
            nn.Conv2d(input_channel, 384, kernel_size=1),
            nn.GELU(),
        )
        
        self.conv_stack_2 = nn.Sequential(
            nn.Conv2d(384, output_channel, kernel_size=1)
        )
        
        # Skip Connection for residual learning
        self.skip_connection = nn.Sequential()
        if input_channel != output_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1),
            )
    
    def forward(self, x):
        """
        Forward pass of the ConvNeXt Block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H, W).
        """
        # Depthwise convolution
        logits = self.d_conv_stack(x)
        
        # Permute for Layer Normalization: [B, C, H, W] -> [B, H, W, C]
        logits = logits.permute(0, 2, 3, 1)
        logits = self.layer_norm(logits)
        # Permute back: [B, H, W, C] -> [B, C, H, W]
        logits = logits.permute(0, 3, 1, 2)
        
        # Pointwise convolutions and activation
        logits = self.conv_stack_1(logits)
        logits = self.conv_stack_2(logits)
        
        # Apply skip connection
        skip = self.skip_connection(x)
        
        return logits + skip

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block.

    This block recalibrates feature maps by modeling channel-wise dependencies. 
    It consists of two main steps: Squeeze (global spatial information embedding) 
    and Excitation (adaptive recalibration of feature maps).

    Args:
        channels (int): Number of input channels.
        reduction (int): Reduction ratio for the hidden layer in the excitation step. Default is 16.

    Example:
        >>> se_block = SEBlock(64)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = se_block(x)

    Shape:
        - Input: (B, C, H, W)
        - Output: (B, C, H, W) [same shape as input]
    """

    def __init__(self, channels, reduction=16):
        """
        Initializes the SEBlock.

        Args:
            channels (int): Number of input channels.
            reduction (int): Reduction ratio for the hidden layer. Default is 16.
        """
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass for the SEBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor with recalibrated channels, same shape as input (B, C, H, W).
        """
        b, c, _, _ = x.size()
        
        # Squeeze step: Global Average Pooling
        squeezed = self.squeeze(x).view(b, c)  # (B, C, 1, 1) -> (B, C)

        # Excitation step: FC layers to learn channel-wise attention
        excitation = self.excitation(squeezed).view(b, c, 1, 1)  # (B, C) -> (B, C, 1, 1)

        # Scale input channels with learned attention weights
        return x * excitation.expand_as(x)

class BottleneckTransformerBlock(nn.Module):
    """
    Bottleneck Transformer Block.

    This block combines convolutional layers with self-attention mechanisms to capture both 
    local and global features, inspired by the Bottleneck Transformer architecture.

    Args:
        input_channel (int): Number of input channels.
        hidden_dim (int): Hidden dimension for the attention mechanism. Default is 64.
        heads (int): Number of attention heads. Default is 4.
        output_channel (int): Number of output channels. Default is 256.

    Example:
        >>> block = BottleneckTransformerBlock(64, 64, 4, 256)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = block(x)

    Shape:
        - Input: (B, C_in, H, W)
        - Output: (B, C_out, H, W)
    """

    def __init__(self, input_channel, hidden_dim=64, heads=4, output_channel=256):
        super(BottleneckTransformerBlock, self).__init__()
        
        # Bottleneck Convolution to reduce dimensions
        self.conv1 = nn.Conv2d(input_channel, hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU()

        # Self-Attention Mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)

        # Bottleneck Convolution to increase dimensions
        self.conv2 = nn.Conv2d(hidden_dim, output_channel, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(output_channel)

        # Skip Connection (adjust dimensions if needed)
        self.skip_connection = nn.Sequential()
        if input_channel != output_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1),
                nn.BatchNorm2d(output_channel),
            )

    def forward(self, x):
        """
        Forward pass of the Bottleneck Transformer Block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H, W).
        """
        # Apply first bottleneck convolution
        logits = self.conv1(x)
        logits = self.bn1(logits)
        logits = self.relu(logits)
        
        # Reshape for attention: [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = logits.shape
        logits = logits.flatten(2).permute(0, 2, 1)

        # Self-Attention
        attn_out, _ = self.attention(logits, logits, logits)
        
        # Reshape back to [B, C, H, W]
        attn_out = attn_out.permute(0, 2, 1).reshape(B, C, H, W)
        
        # Apply second bottleneck convolution
        logits = self.conv2(attn_out)
        logits = self.bn2(logits)

        # Apply skip connection and activation
        skip = self.skip_connection(x)
        return F.relu(logits + skip)

class DenseBlock(nn.Module):
    """
    DenseNet Block.

    Implements a Dense Block from the DenseNet architecture, where each layer receives the 
    concatenated feature maps from all previous layers, promoting feature reuse.

    Args:
        input_channel (int): Number of input channels.
        growth_rate (int): Number of channels added at each layer (k in DenseNet). Default is 32.
        num_layers (int): Number of layers in the dense block. Default is 4.

    Example:
        >>> block = DenseBlock(64, 32, 4)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = block(x)

    Shape:
        - Input: (B, C_in, H, W)
        - Output: (B, C_out, H, W) where C_out = C_in + growth_rate * num_layers
    """

    def __init__(self, input_channel, growth_rate=32, num_layers=4):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        # Create multiple layers, each adding growth_rate channels
        for i in range(num_layers):
            self.layers.append(self._make_layer(input_channel + i * growth_rate, growth_rate))
    
    def _make_layer(self, in_channels, growth_rate):
        """Creates a single layer with BN, ReLU, Conv."""
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        """
        Forward pass of the DenseNet Block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H, W), 
                          where C_out = C_in + growth_rate * num_layers.
        """
        features = [x]  # Store input for concatenation
        
        # Pass through each layer, concatenating outputs
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)

        # Concatenate all features along the channel dimension
        return torch.cat(features, dim=1)

class InceptionBlock(nn.Module):
    """
    Inception Block.

    Implements an Inception Block, inspired by the Inception architecture. This block applies 
    multiple convolutional layers with different kernel sizes in parallel and concatenates their outputs.

    Args:
        input_channel (int): Number of input channels.
        output_channel (int): Number of output channels after concatenation. Default is 256.

    Example:
        >>> block = InceptionBlock(64, 256)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = block(x)

    Shape:
        - Input: (B, C_in, H, W)
        - Output: (B, C_out, H, W) where C_out = output_channel
    """

    def __init__(self, input_channel, output_channel=256):
        super(InceptionBlock, self).__init__()
        
        # 1x1 Convolution Path
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel // 4, kernel_size=1),
            nn.BatchNorm2d(output_channel // 4),
            nn.ReLU()
        )
        
        # 3x3 Convolution Path
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channel // 4),
            nn.ReLU()
        )
        
        # 5x5 Convolution Path
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel // 4, kernel_size=5, padding=2),
            nn.BatchNorm2d(output_channel // 4),
            nn.ReLU()
        )
        
        # Max Pooling Path with 1x1 Convolution
        self.pool_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(input_channel, output_channel // 4, kernel_size=1),
            nn.BatchNorm2d(output_channel // 4),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass of the Inception Block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H, W), 
                          where C_out = output_channel.
        """
        # Apply each path
        path1 = self.conv1x1(x)
        path2 = self.conv3x3(x)
        path3 = self.conv5x5(x)
        path4 = self.pool_path(x)
        
        # Concatenate outputs along the channel dimension
        output = torch.cat([path1, path2, path3, path4], dim=1)
        return output

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
