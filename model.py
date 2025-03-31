import torch
import torch.nn as nn



class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100, spatial=True):
        super(LearnablePositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.spatial = spatial  # Determines if it’s used for spatial or temporal encoding
        
        # Different shapes for spatial vs. temporal embeddings
        if self.spatial:
            self.position_embeddings = nn.Parameter(torch.zeros(1, max_len, 1, d_model))  # (1, 196, 1, D)
        else:
            self.position_embeddings = nn.Parameter(torch.zeros(1, 1, max_len, d_model))  # (1, 1, 16, D)
        
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, x):
        """
        x: (B, 196, T, D) for temporal encoding or (B, T, 196, D) for spatial encoding
        """
        x = x + self.position_embeddings[:, :x.size(1), :, :] if self.spatial else x + self.position_embeddings[:, :, :x.size(2), :]
        return self.dropout(x)



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



"""
- INPUT: (B, T, 3, 224, 224)
- PATCH: (B, T, P, D) where P=224*224/(16*16)=196 patches and D=768 embedding dimension
- CLS: Add CLS token to get (B, T, P+1, D) before applying positional encodings
- SPATIAL: Apply spatial positional encoding, then permute to (B, P+1, T, D) for temporal encoding
- TEMPORAL: Apply temporal encoding, then reshape to (B*T, P+1, D) for spatial attention
- After spatial attention, split CLS tokens and reshape patches to (B, P, T, D) for temporal attention
"""

class TimeSFormer(nn.Module):
    def __init__(self, number_of_frames: int, num_classes: int = 10):
        super(TimeSFormer, self).__init__()

        self._INPUT_DIM = 224
        self._INPUT_CHANNEL = 3
        self._PATCH_SIZE = 16
        self._EMBED_DIM = 768
        self._PATCH_NUM = (224 // self._PATCH_SIZE) ** 2  # 196
        self._FRAME_NUM = number_of_frames
        
        
        # Positional encoding. The +1 in `self._PATCH_NUM + 1` and `self._FRAME_NUM + 1` is for CLS Token
        self.spatial_encoding = LearnablePositionalEmbedding(self._EMBED_DIM, 0.15, self._PATCH_NUM + 1, spatial=True)
        self.temporal_encoding = LearnablePositionalEmbedding(self._EMBED_DIM, 0.15, self._FRAME_NUM + 1, spatial=False)

        # Patch embedding layer (convert patches to embeddings)
        self.patch_embedding = nn.Linear(self._PATCH_SIZE * self._PATCH_SIZE * 3, self._EMBED_DIM)

        
        # Attention
        self.spartial_encoder = TransformerEncoderBlock(self._EMBED_DIM, 8, 1024, 0.1)
        self.temporal_encoder = TransformerEncoderBlock(self._EMBED_DIM, 16, 2048, 0.1)
        
        
        #CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self._EMBED_DIM))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self._EMBED_DIM, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X has the shape of (B, T, 3, 224, 224)
        B, T, C, H, W = x.shape

        if T != self._FRAME_NUM:
            raise ValueError(f"The number of frames in input is {T}, mismatch with the model's expected number of frames {self._FRAME_NUM}")

        if  (C != self._INPUT_CHANNEL) or (H != self._INPUT_DIM) or (W != self._INPUT_DIM):
            raise ValueError(f"The input spartial dimension is ({C}, {H}, {W}), mismatch with the model's expected spartial dimension ({self._INPUT_CHANNEL}, {self._INPUT_DIM}, {self._INPUT_DIM})")
        
        # Extract patches using unfold (B, T, C, H, W) -> (B, T, P, 16*16*3)
        x = x.reshape(B * T, C, H, W)  # Flatten batch & time
        x = x.unfold(2, self._PATCH_SIZE, self._PATCH_SIZE).unfold(3, self._PATCH_SIZE, self._PATCH_SIZE)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B*T, 14, 14, 3, 16, 16)
        x = x.reshape(B, T, self._PATCH_NUM, -1)  # (B, T, P, D)

        # Apply patch embedding
        patch_embeds = self.patch_embedding(x)  # (B, T, P, D)
        
        # Reshape to add CLS token
        patch_embeds = patch_embeds.reshape(B*T, self._PATCH_NUM, -1)  # (B*T, P, D)
        
        # Add CLS tokens for each frame in the batch
        cls_tokens = self.cls_token.expand(B*T, -1, -1)  # (B*T, 1, D)
        tokens = torch.cat([cls_tokens, patch_embeds], dim=1)  # (B*T, P+1, D)
        
        # Reshape back to include time dimension
        tokens = tokens.reshape(B, T, self._PATCH_NUM + 1, -1)  # (B, T, P+1, D)
        
        # Add spatial encoding
        tokens = self.spatial_encoding(tokens)  # (B, T, P+1, D)
        
        # Swap and add temporal encoding
        tokens = tokens.permute(0, 2, 1, 3)  # (B, P+1, T, D)
        tokens = self.temporal_encoding(tokens)  # (B, P+1, T, D)
        
        # Restore to (B, T, P+1, D) shape
        tokens = tokens.permute(0, 2, 1, 3)  # (B, T, P+1, D)
        
        # Reshape for spatial attention
        tokens = tokens.reshape(B*T, self._PATCH_NUM + 1, -1)  # (B*T, P+1, D)
        
        # Apply spatial attention
        spartial_atten = self.spartial_encoder(tokens)  # (B*T, P+1, D)
        
        # Split CLS tokens and patch embeddings
        cls_token_after_spartial = spartial_atten[:, 0:1, :]  # (B*T, 1, D)
        patch_embeds = spartial_atten[:, 1:, :]  # (B*T, P, D)
        
        # Reshape for temporal attention
        patch_embeds = patch_embeds.reshape(B, T, self._PATCH_NUM, -1)  # (B, T, P, D)
        patch_embeds = patch_embeds.permute(0, 2, 1, 3)  # (B, P, T, D)
        
        # Apply temporal attention on each patch position
        patch_embeds = patch_embeds.reshape(B*self._PATCH_NUM, T, -1)  # (B*P, T, D)
        patch_embeds = self.temporal_encoder(patch_embeds)  # (B*P, T, D)
        patch_embeds = patch_embeds.reshape(B, self._PATCH_NUM, T, -1)  # (B, P, T, D)
        
        # Process the CLS token for classification
        # We'll use the spatially-attended CLS token (before temporal)
        cls_token_after_spartial = cls_token_after_spartial.reshape(B, T, self._EMBED_DIM)  # (B, T, D)
        
        # Average the CLS token over time dimension
        global_cls = cls_token_after_spartial.mean(dim=1)  # (B, D)
        
        # Apply classification head
        output = self.classifier(self.dropout(global_cls))  # (B, num_classes)
        
        return output




