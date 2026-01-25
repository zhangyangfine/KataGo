#!/usr/bin/python3
"""
Knowledge Distillation Training Script for KataGo FastViT Models

This script trains a student model (e.g., ft6c96) using knowledge distillation
from a larger teacher model (e.g., b28c512nbt).

Usage:
    python train_distill.py \
        -traindir /path/to/output \
        -datadir /path/to/npz/data \
        -teacher-checkpoint /path/to/teacher.ckpt \
        -student-model ft6c96 \
        -pos-len 19 \
        -batch-size 512

Key features:
- Knowledge distillation from teacher to student
- AdamW optimizer with cosine LR schedule and linear warmup
- EMA model with 0.9995 decay
- Label smoothing for hard targets
- Configurable distillation alpha and temperature
"""

import sys
import os
import argparse
import math
import time
import logging
import json
import datetime
from datetime import timezone
import gc
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from katago.train import modelconfigs
from katago.train.model_pytorch import Model
from katago.train import load_model
from katago.train import data_processing_pytorch
from katago.train.distill_loss import (
    distillation_loss_policy,
    distillation_loss_value,
    distillation_loss_ownership,
    cross_entropy_with_label_smoothing,
    EMAModel,
)


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================

if __name__ == "__main__":
    description = """
    Train a student neural net on Go positions using knowledge distillation
    from a teacher model. Data comes from npz files of batches.
    """

    parser = argparse.ArgumentParser(description=description, add_help=False)
    required_args = parser.add_argument_group('required arguments')
    optional_args = parser.add_argument_group('optional arguments')
    optional_args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                               help='show this help message and exit')

    # Required arguments
    required_args.add_argument('-traindir', help='Dir to write to for recording training results', required=True)
    required_args.add_argument('-datadir', help='Directory with train subdir of npz data', required=True)
    required_args.add_argument('-teacher-checkpoint', help='Path to teacher model checkpoint', required=True)
    required_args.add_argument('-student-model', help='Student model architecture (e.g., ft6c96)', required=True)
    required_args.add_argument('-pos-len', help='Spatial edge length (e.g., 19 for 19x19)', type=int, required=True)
    required_args.add_argument('-batch-size', help='Batch size for training', type=int, required=True)

    # Optional arguments
    optional_args.add_argument('-initial-checkpoint', help='Resume from this student checkpoint', required=False)
    optional_args.add_argument('-samples-per-epoch', help='Samples per epoch', type=int, default=1000000)
    optional_args.add_argument('-max-epochs', help='Maximum epochs to train', type=int, default=300)
    optional_args.add_argument('-warmup-epochs', help='Number of warmup epochs', type=int, default=5)

    # Optimizer arguments
    optional_args.add_argument('-lr', help='Peak learning rate', type=float, default=1e-3)
    optional_args.add_argument('-weight-decay', help='Weight decay for AdamW', type=float, default=0.05)
    optional_args.add_argument('-ema-decay', help='EMA decay rate', type=float, default=0.9995)

    # Distillation arguments
    optional_args.add_argument('-distill-alpha', help='Weight for distillation loss (vs hard labels)', type=float, default=0.5)
    optional_args.add_argument('-temperature', help='Temperature for distillation', type=float, default=4.0)
    optional_args.add_argument('-label-smoothing', help='Label smoothing for hard targets', type=float, default=0.1)

    # Loss weights
    optional_args.add_argument('-policy-weight', help='Policy loss weight', type=float, default=1.0)
    optional_args.add_argument('-value-weight', help='Value loss weight', type=float, default=0.6)
    optional_args.add_argument('-ownership-weight', help='Ownership loss weight', type=float, default=0.015)

    # QAT arguments
    optional_args.add_argument('-enable-qat', help='Enable Quantization Aware Training', action='store_true')
    optional_args.add_argument('-qat-start-epoch', help='Epoch to activate fake quantization', type=int, default=10)
    optional_args.add_argument('-qat-bits', help='Quantization bit width', type=int, default=8)
    optional_args.add_argument('-qat-group-size', help='Group size for per-group quantization', type=int, default=32)

    # Other options
    optional_args.add_argument('-use-teacher-swa', help='Use SWA weights from teacher', action='store_true')
    optional_args.add_argument('-save-every', help='Save checkpoint every N epochs', type=int, default=1)
    optional_args.add_argument('-log-every', help='Log metrics every N batches', type=int, default=100)

    args = vars(parser.parse_args())


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def make_dirs(traindir):
    """Create necessary directories."""
    if not os.path.exists(traindir):
        os.makedirs(traindir)
    longterm_dir = os.path.join(traindir, "longterm_checkpoints")
    if not os.path.exists(longterm_dir):
        os.makedirs(longterm_dir)
    return longterm_dir


def get_checkpoint_path(traindir):
    return os.path.join(traindir, "checkpoint.ckpt")


def detensorify_metrics(metrics):
    """Convert tensor metrics to Python scalars."""
    ret = {}
    for key in metrics:
        if isinstance(metrics[key], torch.Tensor):
            ret[key] = metrics[key].detach().cpu().item()
        else:
            ret[key] = metrics[key]
    return ret


def log_metrics(metrics, metrics_file, train_state):
    """Log metrics to JSON file."""
    metrics["global_step_samples"] = train_state["global_step_samples"]
    metrics["epoch"] = train_state["epoch"]
    metrics["timestamp"] = datetime.datetime.now(timezone.utc).isoformat()
    metrics_file.write(json.dumps(metrics) + "\n")
    metrics_file.flush()


class CosineWarmupScheduler:
    """
    Cosine annealing scheduler with linear warmup.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = lr * (base_lr / self.base_lrs[0])

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lrs[0] * (self.current_step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (1 + math.cos(math.pi * progress))

    def get_last_lr(self):
        return [self.get_lr()]

    def state_dict(self):
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr,
            'base_lrs': self.base_lrs,
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.min_lr = state_dict['min_lr']
        self.base_lrs = state_dict['base_lrs']


# ==============================================================================
# DISTILLATION LOSS COMPUTATION
# ==============================================================================

def compute_distillation_loss(student_outputs, teacher_outputs, batch, args):
    """
    Compute combined distillation loss.

    Returns dict with individual and total losses.
    """
    losses = {}
    device = student_outputs[0][0].device

    # Unpack outputs (main head only, index 0)
    student_out = student_outputs[0]
    teacher_out = teacher_outputs[0]

    # Policy: (N, num_policy_outputs, num_moves+1)
    # We use channel 0 (main policy) for distillation
    student_policy = student_out[0][:, 0, :]  # (N, num_moves+1)
    teacher_policy = teacher_out[0][:, 0, :]  # (N, num_moves+1)

    # Value: (N, 3)
    student_value = student_out[1]
    teacher_value = teacher_out[1]

    # Ownership: (N, 1, H, W)
    student_ownership = student_out[4]
    teacher_ownership = teacher_out[4]

    # ===============================
    # Soft (distillation) losses
    # ===============================
    soft_policy_loss = distillation_loss_policy(
        student_policy, teacher_policy.detach(), args["temperature"]
    )
    losses['soft_policy_loss'] = soft_policy_loss

    soft_value_loss = distillation_loss_value(
        student_value, teacher_value.detach()
    )
    losses['soft_value_loss'] = soft_value_loss

    soft_ownership_loss = distillation_loss_ownership(
        student_ownership, teacher_ownership.detach()
    )
    losses['soft_ownership_loss'] = soft_ownership_loss

    # ===============================
    # Hard (ground truth) losses
    # ===============================
    # Policy target: (N, C, num_moves+1), use channel 0
    policy_target = batch['policyTargetsNCMove'][:, 0, :]  # (N, num_moves+1)
    hard_policy_loss = cross_entropy_with_label_smoothing(
        student_policy, policy_target, args["label_smoothing"]
    )
    losses['hard_policy_loss'] = hard_policy_loss

    # Value target: globalTargetsNC contains value info
    # Indices 0,1,2 are win/loss/draw related
    value_target = batch['globalTargetsNC'][:, 0:3]
    hard_value_loss = cross_entropy_with_label_smoothing(
        student_value, value_target, args["label_smoothing"]
    )
    losses['hard_value_loss'] = hard_value_loss

    # Ownership target: valueTargetsNCHW channel 0
    ownership_target = batch['valueTargetsNCHW'][:, 0:1, :, :]
    hard_ownership_loss = F.mse_loss(student_ownership, ownership_target)
    losses['hard_ownership_loss'] = hard_ownership_loss

    # ===============================
    # Combined losses
    # ===============================
    alpha = args["distill_alpha"]

    policy_loss = alpha * soft_policy_loss + (1.0 - alpha) * hard_policy_loss
    losses['policy_loss'] = policy_loss * args["policy_weight"]

    value_loss = alpha * soft_value_loss + (1.0 - alpha) * hard_value_loss
    losses['value_loss'] = value_loss * args["value_weight"]

    ownership_loss = alpha * soft_ownership_loss + (1.0 - alpha) * hard_ownership_loss
    losses['ownership_loss'] = ownership_loss * args["ownership_weight"]

    # Total loss
    total_loss = losses['policy_loss'] + losses['value_loss'] + losses['ownership_loss']
    losses['total_loss'] = total_loss

    return losses


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def main(args):
    traindir = args["traindir"]
    datadir = args["datadir"]
    teacher_checkpoint = args["teacher_checkpoint"]
    student_model_name = args["student_model"]
    pos_len = args["pos_len"]
    batch_size = args["batch_size"]
    initial_checkpoint = args["initial_checkpoint"]

    samples_per_epoch = args["samples_per_epoch"]
    max_epochs = args["max_epochs"]
    warmup_epochs = args["warmup_epochs"]

    lr = args["lr"]
    weight_decay = args["weight_decay"]
    ema_decay = args["ema_decay"]

    use_teacher_swa = args["use_teacher_swa"]
    save_every = args["save_every"]
    log_every = args["log_every"]

    # Create directories
    longterm_dir = make_dirs(traindir)

    # Setup logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(traindir, "train.log"), mode="a"),
            logging.StreamHandler()
        ],
    )
    logging.info(str(sys.argv))

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS device")
    else:
        device = torch.device("cpu")
        logging.warning("WARNING: No GPU, using CPU")

    # Seed
    seed = int.from_bytes(os.urandom(7), sys.byteorder)
    logging.info(f"Seeding torch with {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))

    # =============================================
    # Load Teacher Model
    # =============================================
    logging.info(f"Loading teacher model from: {teacher_checkpoint}")
    teacher_model, teacher_swa, _ = load_model.load_model(
        teacher_checkpoint,
        use_swa=use_teacher_swa,
        device=device,
        pos_len=pos_len,
        verbose=True
    )
    if use_teacher_swa and teacher_swa is not None:
        teacher_model = teacher_swa
        logging.info("Using SWA weights for teacher")

    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    logging.info("Teacher model loaded and frozen")

    # =============================================
    # Create/Load Student Model
    # =============================================
    student_config = modelconfigs.config_of_name[student_model_name]
    logging.info(f"Student model config: {student_model_name}")
    logging.info(str(student_config))

    student_model = Model(student_config, pos_len)
    student_model.initialize()
    student_model.to(device)

    # Print student model parameters
    total_params = sum(p.numel() for p in student_model.parameters())
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    logging.info(f"Student total params: {total_params:,}")
    logging.info(f"Student trainable params: {trainable_params:,}")

    # Apply QAT if enabled
    enable_qat = args["enable_qat"]
    qat_start_epoch = args["qat_start_epoch"]
    if enable_qat:
        from katago.train.quantize import apply_qat_to_model, set_qat_enabled, count_qat_layers
        n_wrapped = apply_qat_to_model(student_model, bits=args["qat_bits"], group_size=args["qat_group_size"])
        set_qat_enabled(student_model, enabled=False)  # warmup phase
        logging.info(f"QAT: {n_wrapped} layers wrapped, activation at epoch {qat_start_epoch}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay
    )

    # Calculate total steps
    steps_per_epoch = samples_per_epoch // batch_size
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    logging.info(f"Steps per epoch: {steps_per_epoch}")
    logging.info(f"Total steps: {total_steps}")
    logging.info(f"Warmup steps: {warmup_steps}")

    # Create scheduler
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)

    # Create EMA model
    ema_model = EMAModel(student_model, decay=ema_decay)

    # Training state
    train_state = {
        "global_step_samples": 0,
        "epoch": 0,
        "best_val_loss": float('inf'),
    }

    # Load checkpoint if resuming
    checkpoint_path = get_checkpoint_path(traindir)
    if initial_checkpoint is not None and os.path.exists(initial_checkpoint):
        logging.info(f"Loading initial checkpoint: {initial_checkpoint}")
        state_dict = torch.load(initial_checkpoint, map_location=device)
        student_model.load_state_dict(load_model.load_model_state_dict(state_dict))
        if "optimizer" in state_dict:
            optimizer.load_state_dict(state_dict["optimizer"])
        if "scheduler" in state_dict:
            scheduler.load_state_dict(state_dict["scheduler"])
        if "ema_model" in state_dict:
            ema_model.load_state_dict(state_dict["ema_model"])
        if "train_state" in state_dict:
            train_state = state_dict["train_state"]
        logging.info(f"Resumed from epoch {train_state['epoch']}")
    elif os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        student_model.load_state_dict(load_model.load_model_state_dict(state_dict))
        if "optimizer" in state_dict:
            optimizer.load_state_dict(state_dict["optimizer"])
        if "scheduler" in state_dict:
            scheduler.load_state_dict(state_dict["scheduler"])
        if "ema_model" in state_dict:
            ema_model.load_state_dict(state_dict["ema_model"])
        if "train_state" in state_dict:
            train_state = state_dict["train_state"]
        logging.info(f"Resumed from epoch {train_state['epoch']}")

    # =============================================
    # Data Loading
    # =============================================
    import glob as glob_module

    def get_train_files():
        """Get list of training npz files with recursive search."""
        # First try standard layout with train/ subdirectory
        train_subdir = os.path.join(datadir, "train")
        if os.path.exists(train_subdir):
            files = glob_module.glob(os.path.join(train_subdir, "**/*.npz"), recursive=True)
            if files:
                return sorted(files)

        # Try direct directory listing
        if os.path.isdir(datadir):
            # First try flat directory
            files = [os.path.join(datadir, f) for f in os.listdir(datadir) if f.endswith(".npz")]
            if files:
                return sorted(files)

            # Try recursive glob for nested directories
            files = glob_module.glob(os.path.join(datadir, "**/*.npz"), recursive=True)
            if files:
                return sorted(files)

        # Try treating datadir as a glob pattern itself
        files = glob_module.glob(datadir, recursive=True)
        return sorted([f for f in files if f.endswith(".npz")])

    with open(os.path.join(traindir, "metrics_train.json"), "a") as train_metrics_file:
        # =============================================
        # Save function
        # =============================================
        def save_checkpoint(path=None):
            state = {
                "model": student_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema_model": ema_model.state_dict(),
                "train_state": train_state,
                "config": student_config,
            }
            if enable_qat:
                state["qat_config"] = {
                    "bits": args["qat_bits"],
                    "group_size": args["qat_group_size"],
                    "start_epoch": qat_start_epoch,
                }
            if path is None:
                path = checkpoint_path
            logging.info(f"Saving checkpoint to: {path}")
            torch.save(state, path + ".tmp")
            time.sleep(0.5)
            os.replace(path + ".tmp", path)

        # =============================================
        # Training Loop
        # =============================================
        last_longterm_save = datetime.datetime.now()

        logging.info("="*70)
        logging.info("STARTING KNOWLEDGE DISTILLATION TRAINING")
        logging.info("="*70)
        logging.info(f"Teacher: {teacher_checkpoint}")
        logging.info(f"Student: {student_model_name}")
        logging.info(f"Distillation alpha: {args['distill_alpha']}")
        logging.info(f"Temperature: {args['temperature']}")
        logging.info(f"Label smoothing: {args['label_smoothing']}")
        if enable_qat:
            logging.info(f"QAT: bits={args['qat_bits']}, group_size={args['qat_group_size']}, start_epoch={qat_start_epoch}")
        logging.info("="*70)

        while train_state["epoch"] < max_epochs:
            epoch = train_state["epoch"]
            logging.info("="*70)
            logging.info(f"EPOCH {epoch + 1}/{max_epochs}")
            logging.info("="*70)
            logging.info(f"Time: {datetime.datetime.now()}")
            logging.info(f"Global samples: {train_state['global_step_samples']:,}")
            logging.info(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")

            # Toggle QAT at the specified epoch
            if enable_qat:
                qat_active = epoch >= qat_start_epoch
                set_qat_enabled(student_model, enabled=qat_active)
                if qat_active:
                    logging.info(f"QAT: fake quantization ENABLED (epoch {epoch} >= {qat_start_epoch})")

            gc.collect()

            # Get training files
            train_files = get_train_files()
            if len(train_files) == 0:
                logging.error(f"No training files found in {datadir}")
                break
            np.random.shuffle(train_files)
            logging.info(f"Found {len(train_files)} training files")

            # Training
            student_model.train()
            teacher_model.eval()

            running_metrics = defaultdict(float)
            running_count = 0
            batch_count = 0
            epoch_start_time = time.perf_counter()
            last_log_time = time.perf_counter()

            for batch in data_processing_pytorch.read_npz_training_data(
                train_files,
                batch_size,
                world_size=1,
                rank=0,
                pos_len=pos_len,
                device=device,
                randomize_symmetries=True,
                include_meta=student_model.get_has_metadata_encoder(),
                model_config=student_config,
            ):
                optimizer.zero_grad(set_to_none=True)

                # Forward pass
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        batch["binaryInputNCHW"],
                        batch["globalInputNC"],
                    )
                student_outputs = student_model(
                    batch["binaryInputNCHW"],
                    batch["globalInputNC"],
                )

                # Compute loss
                losses = compute_distillation_loss(student_outputs, teacher_outputs, batch, args)
                loss = losses['total_loss']

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()

                # Update scheduler
                scheduler.step()

                # Update EMA
                ema_model.update(student_model)

                # Update counters
                batch_count += 1
                train_state["global_step_samples"] += batch_size
                running_count += batch_size

                # Accumulate metrics
                metrics = detensorify_metrics(losses)
                for key, val in metrics.items():
                    running_metrics[key] += val * batch_size

                # Log periodically
                if batch_count % log_every == 0:
                    t1 = time.perf_counter()
                    time_elapsed = t1 - last_log_time
                    last_log_time = t1

                    avg_metrics = {k: v / running_count for k, v in running_metrics.items()}
                    avg_metrics['lr'] = scheduler.get_last_lr()[0]
                    avg_metrics['samples_per_sec'] = running_count / time_elapsed
                    if enable_qat:
                        avg_metrics['qat_enabled'] = epoch >= qat_start_epoch

                    log_metrics(avg_metrics, train_metrics_file, train_state)

                    logging.info(
                        f"[{batch_count}] loss={avg_metrics['total_loss']:.4f} "
                        f"policy={avg_metrics['policy_loss']:.4f} "
                        f"value={avg_metrics['value_loss']:.4f} "
                        f"lr={avg_metrics['lr']:.2e} "
                        f"samp/s={avg_metrics['samples_per_sec']:.1f}"
                    )

                    running_metrics = defaultdict(float)
                    running_count = 0

                # Stop at samples_per_epoch
                if batch_count * batch_size >= samples_per_epoch:
                    break

            # End of epoch
            epoch_time = time.perf_counter() - epoch_start_time
            logging.info(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")

            train_state["epoch"] += 1

            # Save checkpoint
            if train_state["epoch"] % save_every == 0:
                save_checkpoint()

            # Save longterm checkpoint every 12 hours
            now = datetime.datetime.now()
            if now - last_longterm_save >= datetime.timedelta(hours=12):
                last_longterm_save = now
                dated_name = now.strftime("%Y%m%d-%H%M%S")
                save_checkpoint(os.path.join(longterm_dir, f"{dated_name}.ckpt"))

        # Final save
        save_checkpoint()
        logging.info("Training completed!")


if __name__ == "__main__":
    main(args)
