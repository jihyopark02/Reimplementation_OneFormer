import torch.nn as nn
import torch.nn.functional as F

class MaskClassPredictor(nn.Module):
    def __init__(self, embed_dim, num_queries, num_classes):
        super(MaskClassPredictor, self).__init__()
        # Define layers here
        self.class_proj = nn.Linear(embed_dim, num_classes + 1)
        self.mask_proj = nn.Conv2d(embed_dim, num_queries, kernel_size=1)

    def forward(self, x):
        # Reshape x to expected spatial dimensions before applying mask_proj
        batch_size, seq_len, embed_dim = x.size()  # e.g., [1, 300 + 16384, 256]
        
        # Separate out the sequence dimension and the spatial dimension
        # Assuming the spatial part has 128x128 resolution as per 1/4 multi-scale feature
        spatial_dim = int(seq_len - 300)  # Deducting queries (300) from total sequence
        height = width = int(spatial_dim ** 0.5)
        
        # Reshape x to [batch_size, embed_dim, height, width] for mask prediction
        x_spatial = x[:, 300:, :]  # Select only the spatial part
        x_spatial = x_spatial.permute(0, 2, 1).view(batch_size, embed_dim, height, width)

        # Predict class and mask
        class_pred = self.class_proj(x[:, :300, :].mean(dim=1))  # Only use query part for class prediction
        mask_pred = self.mask_proj(x_spatial)

        return mask_pred, class_pred



