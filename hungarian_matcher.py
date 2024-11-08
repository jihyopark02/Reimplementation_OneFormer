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
        pred_logits = outputs['pred_logits']  # Expected [batch_size, num_queries, num_classes]
        pred_masks = outputs['pred_masks']    # Expected [batch_size, num_queries, H, W]
        target_labels = [t['labels'] for t in targets]  # List of target labels for each batch

        # Only get masks if they exist, otherwise set to None
        target_masks = [t.get('masks', None) for t in targets]

        batch_size = pred_logits.shape[0]
        indices = []

        for i in range(batch_size):
            # Flatten logits and ensure correct shape
            pred_logits_i = pred_logits[i]
            if pred_logits_i.dim() == 1:  # Expand if needed
                pred_logits_i = pred_logits_i.unsqueeze(0)

            # Flatten target labels and remove duplicates
            target_labels_i = target_labels[i].flatten().unique()  # Extract unique class labels
            num_targets = target_labels_i.size(0)

            # Initialize one-hot encoding for classification cost
            target_classes_onehot = torch.zeros((num_targets, pred_logits_i.size(-1)), device=pred_logits.device)
            for j, label in enumerate(target_labels_i):
                if label < pred_logits_i.size(-1):  # Ensure label is within bounds
                    target_classes_onehot[j, label.item()] = 1

            # Compute classification cost
            class_cost = -torch.matmul(pred_logits_i, target_classes_onehot.T)  # [num_queries, num_targets]

            # If there are no target masks, skip mask and dice cost
            if target_masks[i] is None:
                total_cost = self.cost_class * class_cost
            else:
                # Compute costs as usual when masks are available

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





