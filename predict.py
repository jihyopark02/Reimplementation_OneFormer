import torch.nn as nn
import torch.nn.functional as F

class MaskClassPredictor(nn.Module):
    def __init__(self, embed_dim, num_queries, num_classes):
        super(MaskClassPredictor, self).__init__()
        
        # Class prediction layer: [embed_dim -> num_classes + 1]
        self.class_predictor = nn.Linear(embed_dim, num_classes + 1)  # +1 for "no-object" class
        
        # Mask prediction layer: [embed_dim -> mask space]
        self.mask_predictor = nn.Conv2d(embed_dim, num_queries, kernel_size=1)
    
    def forward(self, task_queries, multi_scale_features):
        # Step 1: Class Prediction
        class_logits = self.class_predictor(task_queries)  # Shape: [batch_size, num_queries, num_classes + 1]
        
        # Step 2: Mask Prediction
        # Reshape task queries to match spatial dimensions for mask prediction
        mask_features = task_queries.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, embed_dim, num_queries, 1, 1]
        mask_logits = self.mask_predictor(mask_features).squeeze(2)  # Shape: [batch_size, num_queries, H, W]
        
        return class_logits, mask_logits

