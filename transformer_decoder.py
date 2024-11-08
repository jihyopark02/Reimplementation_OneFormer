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
        # Final linear projection to class logits and mask logits
        self.class_proj = nn.Linear(embed_dim, num_classes + 1)
        self.mask_proj = nn.Conv2d(embed_dim, num_queries, kernel_size=1)

    def forward(self, q_task, multi_scale_features):
        # Verify initial shape of q_task
        print(f"Initial shape of q_task before permute: {q_task.shape}")

        # Ensure q_task is 3D: [batch_size, num_queries, embed_dim]
        if q_task.dim() == 2:  # Possibly missing batch dimension
            q_task = q_task.unsqueeze(0)  # Add batch dimension if missing
            print(f"q_task reshaped to add batch dimension: {q_task.shape}")

        # Stage 1: 1/32 Cross-Attention
        q_task = q_task.permute(1, 0, 2)  # [num_queries, batch_size, embed_dim] for attention
        print(f"Shape of q_task after permute for cross-attention: {q_task.shape}")

        k_v_1_32 = multi_scale_features[3].view(-1, q_task.size(1), q_task.size(2))  # Flatten 1/32 features
        print(f"Shape of k_v_1_32 for cross-attention: {k_v_1_32.shape}")
        q_task, _ = self.cross_attention_1_32(q_task, k_v_1_32, k_v_1_32)
        
        # Stage 2: 1/16 Self-Attention
        k_v_1_16 = multi_scale_features[2].view(-1, q_task.size(1), q_task.size(2))
        print(f"Shape of k_v_1_16 for self-attention: {k_v_1_16.shape}")
        q_task, _ = self.self_attention_1_16(q_task, k_v_1_16, k_v_1_16)

        # Stage 3: 1/8 Feed-Forward Network
        q_task = q_task.permute(1, 0, 2)  # Back to [batch_size, num_queries, embed_dim]
        print(f"Shape of q_task before feed-forward network: {q_task.shape}")
        q_task = self.ffn_1_8(q_task)
        
        return q_task




