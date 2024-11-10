import torch
import torch.nn.functional as F
from torch import nn

class SetCriterion(nn.Module):
    def __init__(self, matcher, num_classes, weight_dict, eos_coef, losses):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.register_buffer("empty_weight", torch.ones(num_classes))

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        idx = self._get_src_permutation_idx(indices)
        
        src_logits = outputs["pred_logits"][idx].float()
        
        if src_logits.dim() == 1:
            src_logits = src_logits.unsqueeze(0)

        target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = target_classes.view(-1)

        if src_logits.shape[0] != target_classes.shape[0]:
            target_classes = target_classes[:src_logits.shape[0]]

        loss_ce = F.cross_entropy(src_logits.float(), target_classes.long(), weight=None)
        print(f"Cross-entropy loss: {loss_ce}")

        if all("masks" in t for t in targets):
            src_masks = outputs["pred_masks"][idx].flatten(1).float()
            target_masks = torch.cat([t["masks"][J] for t, (_, J) in zip(targets, indices)]).flatten(1).float()
            
            loss_mask = F.binary_cross_entropy_with_logits(src_masks, target_masks)
            print(f"Binary cross-entropy loss for masks: {loss_mask}")
        else:
            # Skip mask loss if masks are not available
            loss_mask = torch.tensor(0.0, device=src_logits.device)

        losses = {"loss_ce": loss_ce, "loss_mask": loss_mask}
        
        return losses














