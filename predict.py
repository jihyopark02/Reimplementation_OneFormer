import torch.nn as nn
import torch.nn.functional as F

class MaskClassPredictor(nn.Module):
    def __init__(self, embed_dim, num_queries, num_classes):
        super(MaskClassPredictor, self).__init__()
        self.class_proj = nn.Linear(embed_dim, num_classes + 1)
        self.mask_proj = nn.Conv2d(embed_dim, num_queries, kernel_size=1)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size() 

        spatial_dim = int(seq_len - 300) 
        height = width = int(spatial_dim ** 0.5)
     
        x_spatial = x[:, 300:, :]
        x_spatial = x_spatial.permute(0, 2, 1).view(batch_size, embed_dim, height, width)

        class_pred = self.class_proj(x[:, :300, :].mean(dim=1))
        mask_pred = self.mask_proj(x_spatial)

        return mask_pred, class_pred



