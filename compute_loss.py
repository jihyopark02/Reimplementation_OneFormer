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
        # Permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def forward(self, outputs, targets):
        # Internally call Hungarian matcher for indices
        indices = self.matcher(outputs, targets)
        idx = self._get_src_permutation_idx(indices)
        
        # Extract predicted logits and ensure it has the shape [batch_size, num_classes]
        src_logits = outputs["pred_logits"][idx].float() # fixed hereeeeeeeeeeeeee
        
        if src_logits.dim() == 1:
            src_logits = src_logits.unsqueeze(0)
        print(f"src_logits shape after ensuring 2D: {src_logits.shape}")

        # Prepare target classes and ensure it’s 1D with compatible batch size
        target_classes = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = target_classes.view(-1)  # Flatten to 1D
        print(f"target_classes shape after flattening: {target_classes.shape}")

        # Adjust shapes if necessary to make them compatible
        if src_logits.shape[0] != target_classes.shape[0]:
            print(f"Adjusting target_classes shape to match src_logits shape.")
            target_classes = target_classes[:src_logits.shape[0]]

        print(f"src_logits values:\n{src_logits}")
        print(f"target_classes values:\n{target_classes}")
        print(f"Unique values in target_classes: {target_classes.unique()}")
        print(f"src_logits mean: {src_logits.mean()}, std: {src_logits.std()}")

        # Compute cross-entropy loss
        loss_ce = F.cross_entropy(src_logits.float(), target_classes.long(), weight=None)
        print(f"Cross-entropy loss: {loss_ce}")

        # Compute mask loss
        if all("masks" in t for t in targets):
            src_masks = outputs["pred_masks"][idx].flatten(1).float()
            target_masks = torch.cat([t["masks"][J] for t, (_, J) in zip(targets, indices)]).flatten(1).float()
            print(f"src_masks shape: {src_masks.shape}, target_masks shape: {target_masks.shape}")
            
            loss_mask = F.binary_cross_entropy_with_logits(src_masks, target_masks)
            print(f"Binary cross-entropy loss for masks: {loss_mask}")
        else:
            # Skip mask loss if masks are not available
            loss_mask = torch.tensor(0.0, device=src_logits.device)
            print("No masks found in targets; skipping mask loss calculation.")

        # Aggregate losses
        losses = {"loss_ce": loss_ce, "loss_mask": loss_mask}
        
        return losses














