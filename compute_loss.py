import torch
import torch.nn as nn
import torch.nn.functional as F

class SetCriterion(nn.Module):
    def __init__(self, matcher, num_classes, weight_dict, eos_coef, losses):
        """
        Args:
            num_classes (int): Number of object classes.
            weight_dict (dict): Dictionary specifying the weight for each loss type.
            eos_coef (float): Coefficient for "no-object" class.
            losses (list of str): List of losses to apply (e.g., ["labels", "masks", "dice"]).
        """
        super(SetCriterion, self).__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        # Background weight for "no object" class
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices):
        """
        Classification loss using cross-entropy.
        """
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}

    def loss_masks(self, outputs, targets, indices):
        """
        Mask loss using Binary Cross-Entropy and Dice loss.
        """
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]

        target_masks = torch.cat([t["masks"] for t in targets], dim=0)
        target_masks = target_masks[tgt_idx]

        # Resize masks to the same spatial dimensions
        src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

        # Binary Cross-Entropy Loss
        loss_mask = F.binary_cross_entropy_with_logits(src_masks, target_masks)

        # Dice Loss
        dice_loss = self._dice_loss(src_masks, target_masks)

        return {"loss_mask": loss_mask, "loss_dice": dice_loss}

    def _dice_loss(self, inputs, targets):
        """
        Dice loss calculation.
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        intersection = (inputs * targets).sum(-1)
        union = inputs.sum(-1) + targets.sum(-1)
        dice_loss = 1 - (2 * intersection + 1) / (union + 1)
        return dice_loss.mean()

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices):
        """
        Calls the specific loss computation method.
        """
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f"Loss {loss} not implemented."
        return loss_map[loss](outputs, targets, indices)

    def forward(self, outputs, targets):
        """
        Computes all the losses.
        """
        indices = self.matcher(outputs, targets)
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))
        
        # Apply weights to each loss type
        for k in losses.keys():
            losses[k] *= self.weight_dict.get(k, 1.0)
        
        return losses
