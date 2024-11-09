import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_queries, num_classes, num_heads, num_layers):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.cross_attention_1_32 = nn.MultiheadAttention(embed_dim, num_heads)
        self.self_attention_1_16 = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn_1_8 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.class_proj = nn.Linear(embed_dim, num_classes + 1)
        self.mask_proj = nn.Conv2d(embed_dim, num_queries, kernel_size=1)

    def forward(self, q_task, multi_scale_features):

        if q_task.dim() == 2:  
            q_task = q_task.unsqueeze(0) 

        #1/32 Cross-Attention
        q_task = q_task.permute(1, 0, 2)

        k_v_1_32 = multi_scale_features[3].view(-1, q_task.size(1), q_task.size(2))  
        q_task, _ = self.cross_attention_1_32(q_task, k_v_1_32, k_v_1_32)
        
        #1/16 Self-Attention
        k_v_1_16 = multi_scale_features[2].view(-1, q_task.size(1), q_task.size(2))
        q_task, _ = self.self_attention_1_16(q_task, k_v_1_16, k_v_1_16)

        #1/8 Feed-Forward Network
        q_task = q_task.permute(1, 0, 2)
        q_task = self.ffn_1_8(q_task)
        
        return q_task




