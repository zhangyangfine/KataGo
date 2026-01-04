#!/usr/bin/env python3
"""Run teacher model inference and save soft targets for distillation.

This script loads a trained teacher model checkpoint and runs inference
on generated positions to produce soft targets for student training.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from katago.train.load_model import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_teacher_inference(
    checkpoint_path: str,
    positions_dir: str,
    output_dir: str,
    device: str = "mps",
    batch_size: int = 32,
    use_swa: bool = False,
    pos_len: int = 19,
):
    """Run teacher model inference on generated positions.

    Args:
        checkpoint_path: Path to teacher model checkpoint
        positions_dir: Directory containing position NPZ files
        output_dir: Directory for output target NPZ files
        device: Device to run inference on (mps, cuda, cpu)
        batch_size: Batch size for inference
        use_swa: Whether to use SWA model
        pos_len: Position length for the model
    """
    logging.info(f"Loading teacher model from {checkpoint_path}")

    # Determine device
    if device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning(f"Using CPU device - inference may be slow")

    logging.info(f"Using device: {device}")

    # Load model
    model, swa_model, _ = load_model(
        checkpoint_path,
        use_swa=use_swa,
        device=device,
        pos_len=pos_len,
        verbose=True,
    )

    if use_swa and swa_model is not None:
        model = swa_model
        logging.info("Using SWA model for inference")

    model.eval()

    # Check if model has metadata encoder
    has_metadata = hasattr(model, "metadata_encoder") and model.metadata_encoder is not None
    logging.info(f"Model has metadata encoder: {has_metadata}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each position file
    positions_path = Path(positions_dir)
    position_files = sorted(positions_path.glob("positions_*.npz"))

    if not position_files:
        logging.error(f"No position files found in {positions_dir}")
        return

    logging.info(f"Found {len(position_files)} position files to process")

    for file_idx, npz_file in enumerate(position_files):
        logging.info(f"Processing {npz_file.name} ({file_idx + 1}/{len(position_files)})")

        # Load positions
        with np.load(npz_file) as data:
            binary_input = data["binaryInputNCHW"]
            global_input = data["globalInputNC"]
            if "metadataInputNC" in data and has_metadata:
                meta_input = data["metadataInputNC"]
            else:
                meta_input = None

        num_positions = binary_input.shape[0]
        logging.info(f"  {num_positions} positions")

        # Process in batches
        all_policy = []
        all_value = []
        all_miscvalue = []
        all_moremiscvalue = []
        all_ownership = []
        all_scoring = []
        all_scorebelief = []

        for batch_start in range(0, num_positions, batch_size):
            batch_end = min(batch_start + batch_size, num_positions)

            # Prepare batch
            batch_binary = torch.from_numpy(binary_input[batch_start:batch_end]).to(device)
            batch_global = torch.from_numpy(global_input[batch_start:batch_end]).to(device)

            if meta_input is not None:
                batch_meta = torch.from_numpy(meta_input[batch_start:batch_end]).to(device)
            else:
                batch_meta = None

            # Run inference
            with torch.no_grad():
                outputs = model(
                    input_spatial=batch_binary,
                    input_global=batch_global,
                    input_meta=batch_meta,
                )

            # Extract outputs from main head (index 0)
            # Model returns: ((policy, value, misc, morelisc, ownership, scoring, futurepos, seki, scorebelief),)
            (
                out_policy,
                out_value,
                out_miscvalue,
                out_moremiscvalue,
                out_ownership,
                out_scoring,
                out_futurepos,
                out_seki,
                out_scorebelief,
            ) = outputs[0]

            # Convert to soft targets (probabilities)
            # Policy: softmax over move dimension
            policy_probs = torch.softmax(out_policy, dim=-1)

            # Value: softmax for win/loss/noresult
            value_probs = torch.softmax(out_value, dim=-1)

            # Ownership: tanh activation
            ownership_tanh = torch.tanh(out_ownership)

            # Scoring: tanh activation
            scoring_tanh = torch.tanh(out_scoring)

            # Score belief: already log probabilities, convert to probabilities
            scorebelief_probs = torch.exp(out_scorebelief)

            # Move to CPU and collect
            all_policy.append(policy_probs.cpu().numpy())
            all_value.append(value_probs.cpu().numpy())
            all_miscvalue.append(out_miscvalue.cpu().numpy())
            all_moremiscvalue.append(out_moremiscvalue.cpu().numpy())
            all_ownership.append(ownership_tanh.cpu().numpy())
            all_scoring.append(scoring_tanh.cpu().numpy())
            all_scorebelief.append(scorebelief_probs.cpu().numpy())

            # Sync for MPS
            if device.type == "mps":
                torch.mps.synchronize()

        # Concatenate all batches
        targets = {
            "policy_probs": np.concatenate(all_policy, axis=0).astype(np.float32),
            "value_probs": np.concatenate(all_value, axis=0).astype(np.float32),
            "miscvalue": np.concatenate(all_miscvalue, axis=0).astype(np.float32),
            "moremiscvalue": np.concatenate(all_moremiscvalue, axis=0).astype(np.float32),
            "ownership": np.concatenate(all_ownership, axis=0).astype(np.float32),
            "scoring": np.concatenate(all_scoring, axis=0).astype(np.float32),
            "scorebelief": np.concatenate(all_scorebelief, axis=0).astype(np.float32),
        }

        # Save targets
        output_file = output_path / npz_file.name.replace("positions_", "targets_")
        np.savez_compressed(output_file, **targets)
        logging.info(f"  Saved {output_file}")

    logging.info("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Run teacher model inference for distillation"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to teacher model checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--positions-dir", type=str, required=True,
        help="Directory containing position NPZ files"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for target NPZ files"
    )
    parser.add_argument(
        "--device", type=str, default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device to run inference on (default: mps)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for inference (default: 32)"
    )
    parser.add_argument(
        "--use-swa", action="store_true",
        help="Use SWA model if available"
    )
    parser.add_argument(
        "--pos-len", type=int, default=19,
        help="Position length for model (default: 19)"
    )
    args = parser.parse_args()

    run_teacher_inference(
        checkpoint_path=args.checkpoint,
        positions_dir=args.positions_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        use_swa=args.use_swa,
        pos_len=args.pos_len,
    )


if __name__ == "__main__":
    main()
