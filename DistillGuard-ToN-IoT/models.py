import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridAttention(nn.Module):
    """
    Implements Cross-Attention between Network Features (Query) and Temporal Features (Key/Value).
    This interprets the paper's 'Hybrid Attention' concept.
    """
    def __init__(self, feature_dim, heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_network, x_temporal):
        # x_network: [Batch, Net_Dim] -> projected to [Batch, 1, Hidden]
        # x_temporal: [Batch, Temp_Dim] -> projected to [Batch, 1, Hidden]
        # Attention(Q=Network, K=Temporal, V=Temporal)
        
        # Note: We need to unsqueeze to make them sequence-like [Batch, Seq, Feature]
        # But wait, standard Transformer expects Feature Dimension match.
        # We'll project both inputs to 'feature_dim' first in the main model.
        
        # Assume inputs are already [Batch, 1, feature_dim]
        # query = x_network
        # key = x_temporal
        # value = x_temporal
        
        attn_out, _ = self.attn(x_network, x_temporal, x_temporal)
        x = x_network + self.dropout(attn_out)
        x = self.norm(x)
        return x

class TeacherTransformer(nn.Module):
    def __init__(self, net_input_dim, temp_input_dim, num_classes, hidden_dim=512, layers=6, heads=8, dropout=0.1):
        super().__init__()
        
        # Projection Layers to map varying input sizes to model dimension
        self.net_proj = nn.Linear(net_input_dim, hidden_dim)
        self.temp_proj = nn.Linear(temp_input_dim, hidden_dim)
        
        # Hybrid Attention (Cross-Attention)
        # This layer specifically fuses the two modalities.
        self.hybrid_attn = HybridAttention(hidden_dim, heads, dropout)
        
        # Standard Transformer Encoder Layers (Self-Attention)
        # We apply this AFTER Hybrid Fusion, or in parallel?
        # Paper: "Transformer with hybrid attention: MHSA + Cross-attention".
        # A robust way is:
        # 1. Project inputs.
        # 2. Hybrid Attention (Fuse).
        # 3. Stack of standard Transformer Encoders (Deep processing).
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=heads, dim_feedforward=2048, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x_net, x_temp):
        # Project inputs [Batch, In_Dim] -> [Batch, Hidden]
        net_emb = self.net_proj(x_net).unsqueeze(1)    # [Batch, 1, Hidden]
        temp_emb = self.temp_proj(x_temp).unsqueeze(1) # [Batch, 1, Hidden]
        
        # Hybrid Attention (Cross: Network attends to Temporal)
        # This creates a "network-contextualized-by-temporal" embedding
        fused = self.hybrid_attn(net_emb, temp_emb) # [Batch, 1, Hidden]
        
        # Pass through Deep Transformer Encoder
        out = self.transformer_encoder(fused)       # [Batch, 1, Hidden]
        
        # Flatten and Classify
        out = out.squeeze(1) # [Batch, Hidden]
        logits = self.classifier(out)
        
        return logits, out # Return 'out' (activations) for distillation

class StudentDense(nn.Module):
    def __init__(self, total_input_dim, num_classes, hidden_units=256):
        super().__init__()
        
        # Student takes Concatenated inputs [Network + Temporal]
        self.net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_classes)
        )
        
        # To align with Teacher's 512-dim hidden state for distillation,
        # we might need a projection if we distill intermediate layers.
        # But commonly in KD (like Hinton or SG-KD), we might distill the pen-ultimate layer features.
        # Teacher pen-ultimate is 512. Student pen-ultimate is 256.
        # We add a small adapter to match dimensions for feature-based distillation.
        self.adapter = nn.Linear(hidden_units, 512) 
        
    def forward(self, x):
        # x is concatenated [x_net, x_temp]
        # We need to access intermediate features for distillation
        
        # Let's break down Sequential to get intermediate
        feat = x
        for i, layer in enumerate(self.net[:-1]): # All except last linear
             feat = layer(feat)
             
        # Now feat is at the 3rd ReLU (Size 256)
        logits = self.net[-1](feat)
        
        # Project representation to match Teacher for KD loss
        projected_feat = self.adapter(feat)
        
        return logits, projected_feat
