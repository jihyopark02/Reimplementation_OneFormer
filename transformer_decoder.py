import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_queries, num_classes, hidden_dim=512):
        super(TransformerDecoder, self).__init__()
        
        # Transformer Decoder Layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Linear layers for class and mask prediction
        self.class_predictor = nn.Linear(embed_dim, num_classes + 1)  # +1 for the "no-object" class
        self.mask_predictor = nn.Conv2d(embed_dim, num_queries, kernel_size=1)
    
    def forward(self, multi_scale_features, task_queries):
        # Step 1: Flatten and Concatenate Multi-Scale Features for Transformer Input
        combined_features = torch.cat(
            [f.flatten(2).permute(0, 2, 1) for f in multi_scale_features], dim=1
        )  # Shape: [batch_size, total_spatial_dim, embed_dim]
        
        # Step 2: Decode Task-Conditioned Queries Using Transformer Layers
        repeated_queries = task_queries.unsqueeze(1).repeat(1, combined_features.shape[1], 1, 1)  # [batch_size, total_spatial_dim, num_queries, embed_dim]
        repeated_queries = repeated_queries.flatten(1, 2)

        for layer in self.layers:
            task_queries = layer(repeated_queries, combined_features)
        
        # Step 3: Predict Class Labels from Task Queries
        class_logits = self.class_predictor(task_queries)  # Shape: [batch_size, num_queries, num_classes + 1]

        # Step 4: Predict Masks from Task Queries
        # Reshape task queries to match spatial dimensions for mask prediction
        mask_features = task_queries.permute(0, 2, 1).view(-1, task_queries.size(-1), 1, 1)
        mask_logits = self.mask_predictor(mask_features).squeeze(2).squeeze(2)  # Shape: [batch_size, num_queries, H, W]

        return class_logits, mask_logits

