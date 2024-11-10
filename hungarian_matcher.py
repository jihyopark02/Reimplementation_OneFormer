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
        pred_logits = outputs['pred_logits']  
        pred_masks = outputs['pred_masks']   
        target_labels = [t['labels'] for t in targets] 

        target_masks = [t.get('masks', None) for t in targets]

        batch_size = pred_logits.shape[0]
        indices = []

        for i in range(batch_size):
         
            pred_logits_i = pred_logits[i]
            if pred_logits_i.dim() == 1:
                pred_logits_i = pred_logits_i.unsqueeze(0)

            target_labels_i = target_labels[i].flatten().unique() 
            num_targets = target_labels_i.size(0)

            target_classes_onehot = torch.zeros((num_targets, pred_logits_i.size(-1)), device=pred_logits.device)
            for j, label in enumerate(target_labels_i):
                if label < pred_logits_i.size(-1): 
                    target_classes_onehot[j, label.item()] = 1

            class_cost = -torch.matmul(pred_logits_i, target_classes_onehot.T) 

            if target_masks[i] is None:
                total_cost = self.cost_class * class_cost
            else:
                pred_masks_i = pred_masks[i].flatten(1)  
                target_masks_i = target_masks[i].flatten(1)  
                mask_cost = torch.cdist(pred_masks_i, target_masks_i, p=1)

                pred_masks_i = pred_masks[i].flatten(1).unsqueeze(1)  
                target_masks_i = target_masks[i].flatten(1).unsqueeze(0)  
                intersection = (pred_masks_i * target_masks_i).sum(-1)
                union = pred_masks_i.sum(-1) + target_masks_i.sum(-1)
                dice_cost = 1 - (2 * intersection + 1) / (union + 1)

                total_cost = self.cost_class * class_cost.unsqueeze(-1) + self.cost_mask * mask_cost + self.cost_dice * dice_cost

            row_ind, col_ind = linear_sum_assignment(total_cost.cpu().detach().numpy())
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64), torch.as_tensor(col_ind, dtype=torch.int64)))

        return indices






