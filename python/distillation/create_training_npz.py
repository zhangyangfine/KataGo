#!/usr/bin/env python3
"""Create NPZ training data from positions and teacher targets.

Combines position inputs and teacher soft targets into KataGo's
training NPZ format for student model training.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def pack_binary_input(binary_input: np.ndarray, pos_len: int) -> np.ndarray:
    """Pack binary input features to uint8 format.

    Args:
        binary_input: [N, 22, pos_len, pos_len] float32 array
        pos_len: Position length

    Returns:
        [N, 22, packed_len] uint8 packed array
    """
    n, c, h, w = binary_input.shape
    assert c == 22
    assert h == pos_len and w == pos_len

    # Flatten spatial dimensions
    flat = binary_input.reshape(n, c, -1).astype(np.uint8)  # [N, 22, pos_len*pos_len]

    # Pad to multiple of 8 bits
    pad_len = ((pos_len * pos_len + 7) // 8) * 8
    padded = np.zeros((n, c, pad_len), dtype=np.uint8)
    padded[:, :, :pos_len * pos_len] = flat

    # Pack bits
    packed = np.packbits(padded, axis=2)  # [N, 22, packed_len]

    return packed


def create_global_targets(
    value_probs: np.ndarray,
    miscvalue: np.ndarray,
    pos_len: int,
) -> np.ndarray:
    """Create globalTargetsNC from teacher outputs.

    Args:
        value_probs: [N, 3] win/loss/noresult probabilities
        miscvalue: [N, 10] miscellaneous value outputs
        pos_len: Position length

    Returns:
        [N, 64] global targets array
    """
    n = value_probs.shape[0]
    global_targets = np.zeros((n, 64), dtype=np.float32)

    # Channels 0-2: Value targets (win, loss, noresult)
    global_targets[:, 0:3] = value_probs

    # Channel 3: Score mean (from miscvalue[0])
    global_targets[:, 3] = miscvalue[:, 0]

    # Channels 4-15: TD values (we use same value for all timesteps)
    # Each timestep has 3 value channels + 1 score channel
    for t in range(3):
        base = 4 + t * 4
        global_targets[:, base:base + 3] = value_probs
        global_targets[:, base + 3] = miscvalue[:, 0]  # score

    # Channel 21: Lead (from miscvalue)
    global_targets[:, 21] = miscvalue[:, 1] if miscvalue.shape[1] > 1 else 0.0

    # Channel 22: Variance time
    global_targets[:, 22] = miscvalue[:, 2] if miscvalue.shape[1] > 2 else 0.0

    # Channels 24-35: Weights (all 1.0 for distillation)
    global_targets[:, 24] = 0.0  # td_value weight mask (0 = use)
    global_targets[:, 25] = 1.0  # global_weight
    global_targets[:, 26] = 1.0  # policy player weight
    global_targets[:, 27] = 1.0  # ownership weight
    global_targets[:, 28] = 1.0  # policy opponent weight
    global_targets[:, 29] = 1.0  # lead weight
    global_targets[:, 33] = 1.0  # futurepos weight
    global_targets[:, 34] = 1.0  # scoring weight
    global_targets[:, 35] = 0.0  # value weight mask (0 = use)

    # Channels 36-40: History include flags (all 0 for no history)
    global_targets[:, 36:41] = 0.0

    return global_targets


def create_value_targets(
    ownership: np.ndarray,
    scoring: np.ndarray,
    pos_len: int,
) -> np.ndarray:
    """Create valueTargetsNCHW from teacher outputs.

    Args:
        ownership: [N, 1, H, W] ownership predictions (-1 to 1)
        scoring: [N, 1, H, W] scoring predictions (-1 to 1)
        pos_len: Position length

    Returns:
        [N, 5, pos_len, pos_len] value targets array
    """
    n = ownership.shape[0]
    value_targets = np.zeros((n, 5, pos_len, pos_len), dtype=np.float32)

    # Channel 0: Ownership (-1 to 1)
    value_targets[:, 0, :, :] = ownership[:, 0, :, :]

    # Channel 1: Seki (set to 0)
    value_targets[:, 1, :, :] = 0.0

    # Channels 2-3: Future position (set to 0)
    value_targets[:, 2, :, :] = 0.0
    value_targets[:, 3, :, :] = 0.0

    # Channel 4: Scoring (scaled by 120 as expected by training)
    value_targets[:, 4, :, :] = scoring[:, 0, :, :] * 120.0

    return value_targets


def create_policy_targets(
    policy_probs: np.ndarray,
    pos_len: int,
) -> np.ndarray:
    """Create policyTargetsNCMove from teacher policy outputs.

    Args:
        policy_probs: [N, num_outputs, pos_len*pos_len+1] policy probabilities
        pos_len: Position length

    Returns:
        [N, 2, pos_len*pos_len+1] policy targets array
    """
    n = policy_probs.shape[0]
    num_moves = pos_len * pos_len + 1

    policy_targets = np.zeros((n, 2, num_moves), dtype=np.float32)

    # Channel 0: Player policy (use output 0 from teacher)
    policy_targets[:, 0, :] = policy_probs[:, 0, :]

    # Channel 1: Opponent policy (use output 1 from teacher)
    policy_targets[:, 1, :] = policy_probs[:, 1, :]

    return policy_targets


def create_score_distribution(
    scorebelief: np.ndarray,
) -> np.ndarray:
    """Create score distribution from teacher score belief.

    Args:
        scorebelief: [N, num_buckets] score belief probabilities

    Returns:
        [N, num_buckets] score distribution scaled by 100
    """
    # Scale by 100 as expected by training
    return (scorebelief * 100.0).astype(np.float32)


def create_training_npz(
    positions_dir: str,
    targets_dir: str,
    output_dir: str,
    pos_len: int = 19,
    val_split: float = 0.1,
):
    """Create training NPZ files from positions and teacher targets.

    Args:
        positions_dir: Directory containing position NPZ files
        targets_dir: Directory containing teacher target NPZ files
        output_dir: Output directory for training NPZ files
        pos_len: Position length
        val_split: Fraction of data to use for validation
    """
    positions_path = Path(positions_dir)
    targets_path = Path(targets_dir)
    output_path = Path(output_dir)

    train_path = output_path / "train"
    val_path = output_path / "val"
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    position_files = sorted(positions_path.glob("positions_*.npz"))
    if not position_files:
        logging.error(f"No position files found in {positions_dir}")
        return

    logging.info(f"Found {len(position_files)} position files")

    for file_idx, pos_file in enumerate(position_files):
        # Find matching target file
        target_file = targets_path / pos_file.name.replace("positions_", "targets_")
        if not target_file.exists():
            logging.warning(f"No target file for {pos_file.name}, skipping")
            continue

        logging.info(f"Processing {pos_file.name} ({file_idx + 1}/{len(position_files)})")

        # Load positions
        with np.load(pos_file) as pos_data:
            binary_input = pos_data["binaryInputNCHW"]
            global_input = pos_data["globalInputNC"]
            meta_input = pos_data.get("metadataInputNC", None)

        # Load targets
        with np.load(target_file) as tgt_data:
            policy_probs = tgt_data["policy_probs"]
            value_probs = tgt_data["value_probs"]
            miscvalue = tgt_data["miscvalue"]
            ownership = tgt_data["ownership"]
            scoring = tgt_data["scoring"]
            scorebelief = tgt_data["scorebelief"]

        n_samples = binary_input.shape[0]

        # Create training format arrays
        binary_packed = pack_binary_input(binary_input, pos_len)
        global_targets = create_global_targets(value_probs, miscvalue, pos_len)
        value_targets = create_value_targets(ownership, scoring, pos_len)
        policy_targets = create_policy_targets(policy_probs, pos_len)
        score_distr = create_score_distribution(scorebelief)

        training_data = {
            "binaryInputNCHWPacked": binary_packed,
            "globalInputNC": global_input.astype(np.float32),
            "policyTargetsNCMove": policy_targets,
            "globalTargetsNC": global_targets,
            "scoreDistrN": score_distr,
            "valueTargetsNCHW": value_targets,
        }

        if meta_input is not None:
            training_data["metadataInputNC"] = meta_input.astype(np.float32)

        # Split into train/val
        if file_idx % int(1.0 / val_split) == 0:
            out_file = val_path / f"batch_{file_idx:04d}.npz"
        else:
            out_file = train_path / f"batch_{file_idx:04d}.npz"

        np.savez_compressed(out_file, **training_data)
        logging.info(f"  Saved {out_file} with {n_samples} samples")

        # Create corresponding JSON metadata file for train.py
        json_file = out_file.with_suffix(".json")
        with open(json_file, "w") as f:
            json.dump({"num_rows": n_samples}, f)

    # Create train.json and val.json with total row counts
    for split_name, split_path in [("train", train_path), ("val", val_path)]:
        total_rows = 0
        for npz_file in split_path.glob("*.npz"):
            with np.load(npz_file) as data:
                total_rows += data["globalInputNC"].shape[0]

        split_json = output_path / f"{split_name}.json"
        with open(split_json, "w") as f:
            json.dump({"range": [0, total_rows]}, f)
        logging.info(f"Created {split_json} with range [0, {total_rows}]")

    logging.info("Done creating training NPZ files!")


def main():
    parser = argparse.ArgumentParser(
        description="Create training NPZ files from positions and teacher targets"
    )
    parser.add_argument(
        "--positions-dir", type=str, required=True,
        help="Directory containing position NPZ files"
    )
    parser.add_argument(
        "--targets-dir", type=str, required=True,
        help="Directory containing teacher target NPZ files"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for training NPZ files"
    )
    parser.add_argument(
        "--pos-len", type=int, default=19,
        help="Position length (default: 19)"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1,
        help="Fraction of data for validation (default: 0.1)"
    )
    args = parser.parse_args()

    create_training_npz(
        positions_dir=args.positions_dir,
        targets_dir=args.targets_dir,
        output_dir=args.output_dir,
        pos_len=args.pos_len,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()
