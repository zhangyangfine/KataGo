#!/usr/bin/env python3
"""Build a compact opening book database from KataGo HTML book files.

Parses HTML files via BFS from root, extracts embedded JavaScript data,
and outputs a compact gzipped JSON file for the iOS app.

Usage:
    python scripts/build_book_db.py \
        --book-dir ~/Code/KataGoBooks/book9x9jp-20260226 \
        --output ios/KataGo\ iOS/Resources/book9x9jp.json.gz \
        --av-threshold 1000000
"""

import argparse
import gzip
import json
import os
import re
import sys
from collections import deque

BOARD_SIZE = 9


def parse_html(filepath):
    """Parse a book HTML file and extract the embedded JavaScript data."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        print(f"Warning: cannot read {filepath}: {e}", file=sys.stderr)
        return None

    # Extract nextPla
    m = re.search(r"const nextPla\s*=\s*(\d+);", content)
    if not m:
        return None
    next_pla = int(m.group(1))

    # Extract links: pos -> relative path
    links = {}
    m = re.search(r"const links\s*=\s*\{(.*?)\};", content, re.DOTALL)
    if m:
        for lm in re.finditer(r"(\d+)\s*:\s*'([^']*)'", m.group(1)):
            pos = int(lm.group(1))
            path = lm.group(2)
            if path:
                links[pos] = path

    # Extract linkSyms: pos -> symmetry int
    link_syms = {}
    m = re.search(r"const linkSyms\s*=\s*\{(.*?)\};", content, re.DOTALL)
    if m:
        for sm in re.finditer(r"(\d+)\s*:\s*(\d+)", m.group(1)):
            link_syms[int(sm.group(1))] = int(sm.group(2))

    # Extract moves array
    moves = []
    m = re.search(r"const moves\s*=\s*\[(.*?)\];", content, re.DOTALL)
    if m:
        for mm in re.finditer(r"\{(.*?)\}", m.group(1), re.DOTALL):
            move_data = mm.group(1)
            move = {}

            # Parse xy arrays
            xy_match = re.search(
                r"'xy'\s*:\s*\[((?:\[\d+,\d+\],?\s*)+)\]", move_data
            )
            if xy_match:
                xy = []
                for pair in re.finditer(r"\[(\d+),(\d+)\]", xy_match.group(1)):
                    xy.append((int(pair.group(1)), int(pair.group(2))))
                move["xy"] = xy

            # Parse pass
            if re.search(r"'move'\s*:\s*'pass'", move_data):
                move["pass"] = True

            # Parse numeric fields
            for field in ["p", "wl", "ssM", "v", "av"]:
                fm = re.search(rf"'{field}'\s*:\s*([-\d.eE]+)", move_data)
                if fm:
                    move[field] = float(fm.group(1))

            moves.append(move)

    return {
        "nextPla": next_pla,
        "links": links,
        "linkSyms": link_syms,
        "moves": moves,
    }


def resolve_link_path(current_filepath, link_path):
    """Resolve a relative link path to an absolute filepath."""
    current_dir = os.path.dirname(current_filepath)
    return os.path.normpath(os.path.join(current_dir, link_path))


def build_book(book_dir, av_threshold):
    """BFS through book HTML files, building the position database."""
    root_path = os.path.join(book_dir, "root", "root.html")
    if not os.path.exists(root_path):
        print(f"Error: root file not found at {root_path}", file=sys.stderr)
        sys.exit(1)

    # positions[i] = [nextPla, moves_list, children_list]
    # moves_list = [[pos_list, wl, ss, av, p], ...]
    # children_list = [[pos, childId, sym], ...]
    positions = [None]  # index 0 = root
    path_to_id = {os.path.normpath(root_path): 0}
    queue = deque([(os.path.normpath(root_path), 0)])

    processed = 0

    while queue:
        filepath, pos_id = queue.popleft()

        data = parse_html(filepath)
        if data is None:
            positions[pos_id] = [1, [], []]
            processed += 1
            continue

        # Build set of positions for moves above threshold
        threshold_positions = set()
        move_list = []

        for move in data["moves"]:
            av = move.get("av", 0)
            if av < av_threshold:
                continue

            pos_list = []
            if "xy" in move:
                for x, y in move["xy"]:
                    pos = y * BOARD_SIZE + x
                    pos_list.append(pos)
                    threshold_positions.add(pos)
            elif move.get("pass"):
                pos = BOARD_SIZE * BOARD_SIZE  # 81
                pos_list.append(pos)
                threshold_positions.add(pos)

            wl = round(move.get("wl", 0), 4)
            ss = round(move.get("ssM", 0), 2)
            av_val = int(move.get("av", 0))
            p = round(move.get("p", 0), 4)

            move_list.append([pos_list, wl, ss, av_val, p])

        # Build children list (only for positions meeting threshold)
        children = []
        for pos, link_path in data["links"].items():
            if pos not in threshold_positions:
                continue

            resolved = resolve_link_path(filepath, link_path)
            if not os.path.exists(resolved):
                continue

            if resolved not in path_to_id:
                new_id = len(positions)
                path_to_id[resolved] = new_id
                positions.append(None)
                queue.append((resolved, new_id))

            child_id = path_to_id[resolved]
            link_sym = data["linkSyms"].get(pos, 0)
            children.append([pos, child_id, link_sym])

        positions[pos_id] = [data["nextPla"], move_list, children]

        processed += 1
        if processed % 10000 == 0:
            print(
                f"Processed {processed}/{len(positions)} positions",
                file=sys.stderr,
            )

    return positions


def main():
    parser = argparse.ArgumentParser(
        description="Build compact opening book database from KataGo HTML book files"
    )
    parser.add_argument(
        "--book-dir", required=True, help="Path to KataGo book directory"
    )
    parser.add_argument(
        "--output", required=True, help="Output path for gzipped JSON"
    )
    parser.add_argument(
        "--av-threshold",
        type=int,
        default=1000000,
        help="Minimum adjusted visits to include a move (default: 1000000)",
    )
    args = parser.parse_args()

    print(
        f"Building book from {args.book_dir} (av >= {args.av_threshold})",
        file=sys.stderr,
    )

    positions = build_book(args.book_dir, args.av_threshold)

    # Replace None entries (unreachable positions)
    for i in range(len(positions)):
        if positions[i] is None:
            positions[i] = [1, [], []]

    book = {
        "m": {"s": BOARD_SIZE, "k": 6},
        "p": positions,
    }

    print(f"Total positions: {len(positions)}", file=sys.stderr)

    json_bytes = json.dumps(book, separators=(",", ":")).encode("utf-8")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with gzip.open(args.output, "wb", compresslevel=9) as f:
        f.write(json_bytes)

    compressed_size = os.path.getsize(args.output)
    print(f"Uncompressed: {len(json_bytes) / 1024 / 1024:.1f} MB", file=sys.stderr)
    print(f"Compressed: {compressed_size / 1024 / 1024:.1f} MB", file=sys.stderr)
    print("Done!", file=sys.stderr)


if __name__ == "__main__":
    main()
