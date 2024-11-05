import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_mask=1, cost_dice=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    def forward(self, outputs, targets):
        # Retrieve the output and target tensors for classification, mask, and dice
        pred_logits = outputs['pred_logits']  # [batch_size, num_queries, num_classes]
        pred_masks = outputs['pred_masks']    # [batch_size, num_queries, H, W]
        target_labels = [t['labels'] for t in targets]  # List of target labels for each batch

        # Only get masks if they exist, otherwise set to None
        target_masks = [t.get('masks', None) for t in targets]

        batch_size = pred_logits.shape[0]
        indices = []

        for i in range(batch_size):
            # Flatten the logits for matching
            pred_logits_i = pred_logits[i]  # [num_queries, num_classes]
            target_labels_i = target_labels[i]  # [num_targets]

            # Check if mask exists
            if target_masks[i] is None:
                # Skip mask and dice cost calculation if there is no mask
                # Ensure `target_labels_i` is correctly shaped for multiplication
                if target_labels_i.dim() == 1:
                    target_labels_i = target_labels_i.unsqueeze(0)  # [1, num_targets]
                
                # One-hot encoding for classification cost with compatible shapes
                target_classes_onehot = torch.zeros_like(pred_logits_i)  # [num_queries, num_classes]
                
                # Scatter to get one-hot encoding of target labels
                for j, label in enumerate(target_labels_i.squeeze(0)):
                    target_classes_onehot[j, label] = 1  # Set the correct positions in one-hot format
                
                # Compute classification cost
                class_cost = -torch.matmul(pred_logits_i, target_classes_onehot.T)  # [num_queries, num_targets]
                
                # Use only the classification cost when no mask is available
                total_cost = self.cost_class * class_cost
            else:
                # Compute costs as usual when masks are available

                # Ensure one-hot encoding of target labels is compatible with pred_logits_i
                target_classes_onehot = torch.zeros_like(pred_logits_i)  # [num_queries, num_classes]
                for j, label in enumerate(target_labels_i):
                    target_classes_onehot[j, label] = 1  # Populate one-hot encoding for each target

                # Compute classification cost
                class_cost = -torch.einsum("qc,qc->q", pred_logits_i, target_classes_onehot)

                # Mask cost computation
                pred_masks_i = pred_masks[i].flatten(1)  # [num_queries, H*W]
                target_masks_i = target_masks[i].flatten(1)  # [num_targets, H*W]
                mask_cost = torch.cdist(pred_masks_i, target_masks_i, p=1)

                # Dice cost computation
                pred_masks_i = pred_masks[i].flatten(1).unsqueeze(1)  # [num_queries, 1, H*W]
                target_masks_i = target_masks[i].flatten(1).unsqueeze(0)  # [1, num_targets, H*W]
                intersection = (pred_masks_i * target_masks_i).sum(-1)
                union = pred_masks_i.sum(-1) + target_masks_i.sum(-1)
                dice_cost = 1 - (2 * intersection + 1) / (union + 1)

                # Combine costs
                total_cost = self.cost_class * class_cost.unsqueeze(-1) + self.cost_mask * mask_cost + self.cost_dice * dice_cost

        # Perform Hungarian matching
        row_ind, col_ind = linear_sum_assignment(total_cost.cpu().detach().numpy())
        indices.append((torch.as_tensor(row_ind, dtype=torch.int64), torch.as_tensor(col_ind, dtype=torch.int64)))


        return indices


