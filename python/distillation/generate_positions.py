#!/usr/bin/env python3
"""Generate random Go positions for knowledge distillation training.

This script generates random legal Go positions and converts them to
KataGo's input format (22 binary features, 19 global features, 192 metadata features).
"""

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from katago.game.board import Board
from katago.game.sgfmetadata import SGFMetadata
from katago.train import modelconfigs


def create_input_features(board: Board, next_player: int, pos_len: int = 19):
    """Convert board state to KataGo input features.

    Args:
        board: Board object with current game state
        next_player: Board.BLACK or Board.WHITE
        pos_len: Position length (board size for network)

    Returns:
        binaryInputNCHW: [22, pos_len, pos_len] float32 array
        globalInputNC: [19] float32 array
    """
    opp = Board.get_opp(next_player)
    x_size = board.x_size
    y_size = board.y_size

    # Binary input features - 22 channels
    binary_input = np.zeros((22, pos_len, pos_len), dtype=np.float32)

    for y in range(y_size):
        for x in range(x_size):
            loc = board.loc(x, y)
            stone = board.board[loc]

            # Channel 0: Board mask (valid positions)
            binary_input[0, y, x] = 1.0

            # Channel 1: Current player's stones
            if stone == next_player:
                binary_input[1, y, x] = 1.0
            # Channel 2: Opponent's stones
            elif stone == opp:
                binary_input[2, y, x] = 1.0

            # Channels 3-5: Liberty counts for stones
            if stone == next_player or stone == opp:
                libs = board.num_liberties(loc)
                if libs == 1:
                    binary_input[3, y, x] = 1.0
                elif libs == 2:
                    binary_input[4, y, x] = 1.0
                elif libs == 3:
                    binary_input[5, y, x] = 1.0

    # Channel 6: Simple ko point
    if board.simple_ko_point is not None:
        ko_x = board.loc_x(board.simple_ko_point)
        ko_y = board.loc_y(board.simple_ko_point)
        if 0 <= ko_x < x_size and 0 <= ko_y < y_size:
            binary_input[6, ko_y, ko_x] = 1.0

    # Channels 7-13: Previous move locations (left blank for random positions)
    # Channels 14-17: Ladder features (left blank for simplicity)
    # Channels 18-19: Area/territory features (left blank)
    # Channels 20-21: Encore phase features (left blank)

    # Global input features - 19 channels
    global_input = np.zeros(19, dtype=np.float32)

    # Channels 0-4: Previous pass moves (all 0 for random positions)

    # Channel 5: Self komi (normalized)
    # Default to 7.5 komi for Japanese rules
    white_komi = 7.5
    self_komi = white_komi if next_player == Board.WHITE else -white_komi
    b_area = x_size * y_size
    if self_komi > b_area + 1:
        self_komi = b_area + 1
    if self_komi < -b_area - 1:
        self_komi = -b_area - 1
    global_input[5] = self_komi / 20.0

    # Channels 6-7: Ko rule features (default simple ko)
    # Channels 8: Multi-stone suicide (0 = not legal)
    # Channels 9-11: Scoring and tax rules (default area scoring, no tax)
    # Channels 12-14: Encore phase (0 = normal play)
    # Channels 15-16: Asymmetric rules (0)
    # Channel 17: Button (0)
    # Channel 18: Komi wave (0 for standard komi)

    return binary_input, global_input


def generate_random_position(board_size: int = 19, rand: random.Random = None):
    """Generate a random legal Go position.

    Uses exponential distribution for number of moves to get varied positions
    from opening to endgame.

    Args:
        board_size: Size of the Go board
        rand: Random number generator

    Returns:
        board: Board object with random position
        next_player: Board.BLACK or Board.WHITE
    """
    if rand is None:
        rand = random.Random()

    board = Board(board_size)

    # Random number of moves using exponential distribution
    def randint_exponential(scale):
        r = 0
        while r <= 0:
            r = rand.random()
        return int(math.floor(-math.log(r) * scale))

    both_plays = 1 + randint_exponential(5) + randint_exponential(5) + randint_exponential(12)
    extra_b_plays = randint_exponential(1.5)
    extra_w_plays = randint_exponential(1)

    plays = []
    for _ in range(both_plays + extra_b_plays):
        plays.append(Board.BLACK)
    for _ in range(both_plays + extra_w_plays):
        plays.append(Board.WHITE)
    rand.shuffle(plays)

    # Limit max moves
    if len(plays) > 200:
        plays = plays[:200]

    for pla in plays:
        # Generate weighted random move (prefer center over edges)
        choices = []
        weights = []
        for y in range(board_size):
            for x in range(board_size):
                line = min(y + 1, x + 1, board_size - y, board_size - x)
                if line <= 1:
                    relprob = 1
                elif line <= 2:
                    relprob = 4
                else:
                    relprob = 20
                choices.append((x, y))
                weights.append(relprob)

        (x, y) = rand.choices(choices, weights=weights, k=1)[0]
        loc = board.loc(x, y)

        if board.would_be_legal(pla, loc):
            try:
                board.play(pla, loc)
            except Exception:
                pass

    # Random next player
    next_player = rand.choice([Board.BLACK, Board.WHITE])

    return board, next_player


def generate_positions_batch(
    num_positions: int,
    board_size: int = 19,
    pos_len: int = 19,
    seed: int = None,
    include_human_metadata: bool = True,
):
    """Generate a batch of random positions with input features.

    Args:
        num_positions: Number of positions to generate
        board_size: Size of Go board
        pos_len: Position length for network input
        seed: Random seed
        include_human_metadata: Whether to generate human metadata features

    Returns:
        dict with:
            binaryInputNCHW: [N, 22, pos_len, pos_len] float32
            globalInputNC: [N, 19] float32
            metadataInputNC: [N, 192] float32 (if include_human_metadata)
    """
    rand = random.Random(seed)

    binary_inputs = []
    global_inputs = []
    metadata_inputs = []

    for i in range(num_positions):
        if i % 1000 == 0 and i > 0:
            print(f"Generated {i}/{num_positions} positions")

        board, next_player = generate_random_position(board_size, rand)
        binary_input, global_input = create_input_features(board, next_player, pos_len)

        binary_inputs.append(binary_input)
        global_inputs.append(global_input)

        if include_human_metadata:
            # Generate random human metadata
            metadata = SGFMetadata.get_katago_selfplay_metadata(rand)
            # Make it look like a human game
            if rand.random() < 0.5:
                metadata.bIsHuman = True
                metadata.wIsHuman = True
                # Random rank (0-30 for 30k to 1d)
                metadata.inverseBRank = rand.randint(0, 30)
                metadata.inverseWRank = rand.randint(0, 30)
            meta_row = metadata.get_metadata_row(next_player, board_size * board_size)
            metadata_inputs.append(meta_row)

    result = {
        "binaryInputNCHW": np.stack(binary_inputs).astype(np.float32),
        "globalInputNC": np.stack(global_inputs).astype(np.float32),
    }

    if include_human_metadata:
        result["metadataInputNC"] = np.stack(metadata_inputs).astype(np.float32)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate random Go positions for distillation training"
    )
    parser.add_argument(
        "--num-positions", type=int, default=10000,
        help="Number of positions to generate"
    )
    parser.add_argument(
        "--board-size", type=int, default=19,
        help="Board size (default: 19)"
    )
    parser.add_argument(
        "--pos-len", type=int, default=19,
        help="Position length for network (default: 19)"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for NPZ files"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024,
        help="Number of positions per NPZ file"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--no-metadata", action="store_true",
        help="Don't include human metadata features"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_positions} positions...")
    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Output directory: {output_dir}")

    # Generate positions in batches
    batch_idx = 0
    remaining = args.num_positions
    seed = args.seed

    while remaining > 0:
        batch_num = min(args.batch_size, remaining)

        positions = generate_positions_batch(
            num_positions=batch_num,
            board_size=args.board_size,
            pos_len=args.pos_len,
            seed=seed,
            include_human_metadata=not args.no_metadata,
        )

        output_file = output_dir / f"positions_{batch_idx:04d}.npz"
        np.savez_compressed(output_file, **positions)
        print(f"Saved {output_file} with {batch_num} positions")

        batch_idx += 1
        remaining -= batch_num
        if seed is not None:
            seed += 1

    print(f"Done! Generated {args.num_positions} positions in {batch_idx} files")


if __name__ == "__main__":
    main()
