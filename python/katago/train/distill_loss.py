"""
Distillation loss functions for training student models with knowledge distillation from teacher models.

This module provides loss functions for:
- Policy distillation using KL divergence with temperature scaling
- Value distillation using MSE on softmax outputs
- Label smoothing for hard targets
- Combined distillation loss with configurable alpha weighting
"""

import torch
import torch.nn.functional as F


def distillation_loss_policy(student_logits: torch.Tensor,
                              teacher_logits: torch.Tensor,
                              temperature: float = 4.0) -> torch.Tensor:
    """
    Compute policy distillation loss using KL divergence with temperature scaling.

    The temperature parameter softens the probability distribution, making it easier
    for the student to learn from the teacher's knowledge about relative move quality.

    Args:
        student_logits: Student model policy logits, shape (N, num_moves) or (N, C, num_moves)
        teacher_logits: Teacher model policy logits, shape (N, num_moves) or (N, C, num_moves)
        temperature: Temperature for softening distributions (higher = softer)

    Returns:
        Scalar loss value (KL divergence scaled by temperature^2)
    """
    # Guard against empty batch to prevent NaN from batchmean reduction
    if student_logits.shape[0] == 0:
        return torch.tensor(0.0, device=student_logits.device)

    # Apply temperature scaling
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # KL divergence: sum over classes, mean over batch
    # Scale by T^2 as per Hinton et al. (2015) to match gradient magnitudes
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

    return kl_loss


def distillation_loss_value(student_logits: torch.Tensor,
                            teacher_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute value distillation loss using MSE on softmax outputs.

    For value heads, we match the soft probabilities directly using MSE,
    which works well for the 3-class (win/loss/draw) value distribution.

    Args:
        student_logits: Student model value logits, shape (N, num_outcomes)
        teacher_logits: Teacher model value logits, shape (N, num_outcomes)

    Returns:
        Scalar MSE loss value
    """
    student_probs = F.softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)

    return F.mse_loss(student_probs, teacher_probs)


def distillation_loss_ownership(student_logits: torch.Tensor,
                                 teacher_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute ownership distillation loss using MSE.

    Ownership predictions are per-position values in range [-1, 1],
    so we use MSE directly.

    Args:
        student_logits: Student ownership predictions, shape (N, 1, H, W)
        teacher_logits: Teacher ownership predictions, shape (N, 1, H, W)

    Returns:
        Scalar MSE loss value
    """
    return F.mse_loss(student_logits, teacher_logits)


def distillation_loss_features(
    student_features: list,
    teacher_features: list,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute feature distillation loss using MSE on normalized intermediate feature maps.

    Args:
        student_features: List of student intermediate feature tensors
        teacher_features: List of teacher intermediate feature tensors
        device: Device to place the result tensor on (for empty feature case)

    Returns:
        Scalar MSE loss averaged over all feature pairs
    """
    if not student_features or not teacher_features:
        return torch.tensor(0.0, device=device)

    total_loss = 0.0
    for s_feat, t_feat in zip(student_features, teacher_features):
        # Normalize and compute MSE
        s_norm = F.normalize(s_feat.flatten(2), dim=-1)
        t_norm = F.normalize(t_feat.detach().flatten(2), dim=-1)
        total_loss += F.mse_loss(s_norm, t_norm)
    return total_loss / len(student_features)


def cross_entropy_with_label_smoothing(pred_logits: torch.Tensor,
                                        target: torch.Tensor,
                                        smoothing: float = 0.1) -> torch.Tensor:
    """
    Compute cross entropy loss with label smoothing.

    Label smoothing helps prevent the model from becoming overconfident
    and improves generalization.

    Args:
        pred_logits: Predicted logits, shape (N, num_classes) or (N, C, num_classes)
        target: Target probabilities or one-hot, shape (N, num_classes) or (N, C, num_classes)
        smoothing: Label smoothing factor (0.0 = no smoothing, 1.0 = uniform)

    Returns:
        Scalar cross entropy loss with smoothing
    """
    n_classes = pred_logits.shape[-1]

    # Apply label smoothing: target * (1 - smoothing) + smoothing / n_classes
    smoothed_target = target * (1.0 - smoothing) + smoothing / n_classes

    # Cross entropy with soft targets
    log_probs = F.log_softmax(pred_logits, dim=-1)
    loss = -torch.sum(smoothed_target * log_probs, dim=-1)

    return loss.mean()


def combined_distillation_loss(student_outputs: dict,
                               teacher_outputs: dict,
                               batch: dict,
                               alpha: float = 0.5,
                               temperature: float = 4.0,
                               label_smoothing: float = 0.1,
                               policy_weight: float = 1.0,
                               value_weight: float = 0.6,
                               ownership_weight: float = 0.015) -> dict:
    """
    Compute combined distillation loss with both soft (teacher) and hard (ground truth) targets.

    Total loss = alpha * soft_loss + (1 - alpha) * hard_loss

    Where soft_loss uses teacher predictions and hard_loss uses ground truth labels.

    Args:
        student_outputs: Dictionary with student model outputs
            - 'policy': Policy logits (N, num_moves)
            - 'value': Value logits (N, 3)
            - 'ownership': Ownership predictions (N, 1, H, W)
        teacher_outputs: Dictionary with teacher model outputs (same structure)
        batch: Dictionary with ground truth targets
            - 'policyTargetsNCMove': Policy targets (N, 1, num_moves+1) or similar
            - 'globalTargetsNC': Global targets including value
            - 'valueTargetsNCHW': Ownership targets
        alpha: Weight for distillation loss (1.0 = pure distillation, 0.0 = pure hard labels)
        temperature: Temperature for policy distillation
        label_smoothing: Label smoothing factor for hard targets
        policy_weight: Weight for policy loss
        value_weight: Weight for value loss
        ownership_weight: Weight for ownership loss

    Returns:
        Dictionary with individual and total losses
    """
    losses = {}

    # ===============================
    # Policy Loss
    # ===============================
    student_policy = student_outputs['policy']
    teacher_policy = teacher_outputs['policy']

    # Soft policy loss (distillation from teacher)
    soft_policy_loss = distillation_loss_policy(student_policy, teacher_policy, temperature)
    losses['soft_policy_loss'] = soft_policy_loss

    # Hard policy loss (from ground truth)
    # policyTargetsNCMove has shape (N, C, num_moves+1), we take channel 0 and exclude pass
    policy_target = batch['policyTargetsNCMove'][:, 0, :]  # (N, num_moves+1)
    hard_policy_loss = cross_entropy_with_label_smoothing(student_policy, policy_target, label_smoothing)
    losses['hard_policy_loss'] = hard_policy_loss

    # Combined policy loss
    policy_loss = alpha * soft_policy_loss + (1.0 - alpha) * hard_policy_loss
    losses['policy_loss'] = policy_loss * policy_weight

    # ===============================
    # Value Loss
    # ===============================
    student_value = student_outputs['value']
    teacher_value = teacher_outputs['value']

    # Soft value loss (distillation from teacher)
    soft_value_loss = distillation_loss_value(student_value, teacher_value)
    losses['soft_value_loss'] = soft_value_loss

    # Hard value loss (from ground truth)
    # globalTargetsNC contains value targets at specific indices
    # Index 0-2 are typically win/loss/draw or similar
    value_target = batch['globalTargetsNC'][:, 0:3]  # (N, 3)
    hard_value_loss = cross_entropy_with_label_smoothing(student_value, value_target, label_smoothing)
    losses['hard_value_loss'] = hard_value_loss

    # Combined value loss
    value_loss = alpha * soft_value_loss + (1.0 - alpha) * hard_value_loss
    losses['value_loss'] = value_loss * value_weight

    # ===============================
    # Ownership Loss
    # ===============================
    if 'ownership' in student_outputs and 'ownership' in teacher_outputs:
        student_ownership = student_outputs['ownership']
        teacher_ownership = teacher_outputs['ownership']

        # Soft ownership loss (distillation from teacher)
        soft_ownership_loss = distillation_loss_ownership(student_ownership, teacher_ownership)
        losses['soft_ownership_loss'] = soft_ownership_loss

        # Hard ownership loss (from ground truth)
        ownership_target = batch['valueTargetsNCHW'][:, 0:1, :, :]  # (N, 1, H, W)
        hard_ownership_loss = F.mse_loss(student_ownership, ownership_target)
        losses['hard_ownership_loss'] = hard_ownership_loss

        # Combined ownership loss
        ownership_loss = alpha * soft_ownership_loss + (1.0 - alpha) * hard_ownership_loss
        losses['ownership_loss'] = ownership_loss * ownership_weight
    else:
        losses['ownership_loss'] = torch.tensor(0.0, device=student_policy.device)

    # ===============================
    # Total Loss
    # ===============================
    total_loss = losses['policy_loss'] + losses['value_loss'] + losses['ownership_loss']
    losses['total_loss'] = total_loss

    return losses


class EMAModel:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy of model parameters that is updated with EMA.
    This provides a smoothed version of the model for evaluation.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9995):
        """
        Initialize EMA model.

        Args:
            model: PyTorch model to track
            decay: EMA decay rate (higher = slower update, more smoothing)
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

        # Initialize shadow buffers (for BatchNorm running stats)
        self.shadow_buffers = {}
        for name, buffer in model.named_buffers():
            if 'running_mean' in name or 'running_var' in name:
                self.shadow_buffers[name] = buffer.data.clone()
        self.backup_buffers = {}

    def update(self, model: torch.nn.Module):
        """
        Update shadow parameters with current model parameters.

        shadow = decay * shadow + (1 - decay) * param

        Args:
            model: Model with updated parameters
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )

        # Update shadow buffers (BatchNorm running stats)
        for name, buffer in model.named_buffers():
            if name in self.shadow_buffers:
                self.shadow_buffers[name] = (
                    self.decay * self.shadow_buffers[name] +
                    (1.0 - self.decay) * buffer.data
                )

    def apply_shadow(self, model: torch.nn.Module):
        """
        Apply shadow parameters to model (for evaluation).
        Backs up current parameters first.

        Args:
            model: Model to apply shadow parameters to
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

        # Apply shadow buffers (BatchNorm running stats)
        for name, buffer in model.named_buffers():
            if name in self.shadow_buffers:
                self.backup_buffers[name] = buffer.data.clone()
                buffer.data = self.shadow_buffers[name].clone()

    def restore(self, model: torch.nn.Module):
        """
        Restore original parameters to model (after evaluation).

        Args:
            model: Model to restore original parameters to
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}

        # Restore buffers (BatchNorm running stats)
        for name, buffer in model.named_buffers():
            if name in self.backup_buffers:
                buffer.data = self.backup_buffers[name].clone()
        self.backup_buffers = {}

    def state_dict(self) -> dict:
        """Return state dict for saving."""
        return {
            'decay': self.decay,
            'shadow': self.shadow.copy(),
            'shadow_buffers': self.shadow_buffers.copy(),
        }

    def load_state_dict(self, state_dict: dict):
        """Load state dict for restoring."""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow'].copy()
        # Backward compatibility: old checkpoints may not have shadow_buffers
        self.shadow_buffers = state_dict.get('shadow_buffers', {}).copy()
