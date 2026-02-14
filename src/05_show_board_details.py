import argparse
import csv
import logging
import os
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from board_helpers import (
    K_BLOCKERS,
    NUM_SQUARES,
    SYM_MAPS,
    PIECE_NAMES,
    iter_set_bits,
    indices_to_bitmask,
    bitmask_to_coords,
    build_cover_map,
    canonical_bitmask_from_indices,
    classify_by_parity,
    count_solutions,
    full_board_mask,
    index_to_row_col,
    iter_solutions,
)
from path_helpers import DATA_DIR, OUTPUT_DIR, ensure_output_dir

LOGGER = logging.getLogger(__name__)

SYM_BIT_MAPS = [[1 << mapped_index for mapped_index in symmetry_map] for symmetry_map in SYM_MAPS]


# =========================
# Coord parsing
# =========================
def coord_to_i(coord: str) -> int:
    """Convert a coordinate like 'a1' into a 0-based cell index."""
    coord = coord.strip().lower()
    if len(coord) < 2:
        raise ValueError(f"invalid coord: {coord}")
    row_char = coord[0]
    if row_char < "a" or row_char > "f":
        raise ValueError(f"invalid coord row: {coord}")
    col_str = coord[1:]
    if not col_str.isdigit():
        raise ValueError(f"invalid coord col: {coord}")
    col = int(col_str)
    if col < 1 or col > 6:
        raise ValueError(f"invalid coord col: {coord}")
    row = ord(row_char) - ord("a")
    col_index = col - 1
    return row * 6 + col_index


def parse_coords(coords: List[str]) -> List[int]:
    """Parse and validate exactly K_BLOCKERS coordinates."""
    indices = [coord_to_i(coord) for coord in coords]
    if len(indices) != K_BLOCKERS:
        raise ValueError(f"expected {K_BLOCKERS} coords, got {len(indices)}")
    if len(set(indices)) != K_BLOCKERS:
        raise ValueError("duplicate coordinates provided")
    return indices


# =========================
# Helpers
# =========================
def symmetry_masks(indices: List[int]) -> List[int]:
    """Return all unique masks for the given indices under D4 symmetry."""
    masks: List[int] = []
    for symmetry_bit_map in SYM_BIT_MAPS:
        mask = 0
        for index in indices:
            mask |= symmetry_bit_map[index]
        masks.append(mask)
    return sorted(set(masks))


def render_bitmask_image(mask: int, out_path: str, cell_size: int = 80, margin: int = 20) -> None:
    """Render a board image with blockers for a bitmask."""
    board_size = cell_size * 6
    img_size = board_size + margin * 2
    img = np.full((img_size, img_size, 3), 255, dtype=np.uint8)

    # Draw grid
    for i in range(7):
        x = margin + i * cell_size
        y = margin + i * cell_size
        cv2.line(img, (x, margin), (x, margin + board_size), (0, 0, 0), 2)
        cv2.line(img, (margin, y), (margin + board_size, y), (0, 0, 0), 2)

    # Draw blockers
    radius = int(cell_size * 0.35)
    for i in iter_set_bits(mask):
        r, c = index_to_row_col(i)
        cx = margin + int((c + 0.5) * cell_size)
        cy = margin + int((r + 0.5) * cell_size)
        cv2.circle(img, (cx, cy), radius, (0, 0, 0), -1)

    cv2.imwrite(out_path, img)


def render_solution_image(
    blocked_mask: int,
    placement_by_piece: Dict[str, int],
    out_path: str,
    cell_size: int = 80,
    margin: int = 20,
) -> None:
    """Render a board image with blockers and a solution placement."""
    board_size = cell_size * 6
    img_size = board_size + margin * 2
    img = np.full((img_size, img_size, 3), 255, dtype=np.uint8)

    # Draw grid
    for i in range(7):
        x = margin + i * cell_size
        y = margin + i * cell_size
        cv2.line(img, (x, margin), (x, margin + board_size), (0, 0, 0), 2)
        cv2.line(img, (margin, y), (margin + board_size, y), (0, 0, 0), 2)

    # Fill piece placements
    palette = [
        (231, 76, 60),
        (46, 204, 113),
        (52, 152, 219),
        (155, 89, 182),
        (241, 196, 15),
        (230, 126, 34),
        (26, 188, 156),
        (149, 165, 166),
        (52, 73, 94),
    ]
    for idx, piece in enumerate(PIECE_NAMES):
        placement_mask = placement_by_piece.get(piece)
        if placement_mask is None:
            continue
        color = palette[idx % len(palette)]
        for cell_index in iter_set_bits(placement_mask):
            row, col = index_to_row_col(cell_index)
            x1 = margin + col * cell_size + 2
            y1 = margin + row * cell_size + 2
            x2 = margin + (col + 1) * cell_size - 2
            y2 = margin + (row + 1) * cell_size - 2
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    # Draw blockers on top
    radius = int(cell_size * 0.35)
    for cell_index in iter_set_bits(blocked_mask):
        row, col = index_to_row_col(cell_index)
        cx = margin + int((col + 0.5) * cell_size)
        cy = margin + int((row + 0.5) * cell_size)
        cv2.circle(img, (cx, cy), radius, (0, 0, 0), -1)

    cv2.imwrite(out_path, img)


def find_unique_row_by_canonical_bitmask(csv_path: str, canon_bitmask: int) -> Optional[Dict[str, str]]:
    """Find a row in the unique combos CSV by canonical bitmask."""
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["canonical_bitmask"]) == canon_bitmask:
                return row
    return None


def print_solution_tree(
    blocked_mask: int,
    depth_limit: int,
    max_nodes: int,
    max_children: int,
) -> None:
    """Print a partial DFS solution tree for the given blocker mask."""
    cover_map = build_cover_map(blocked_mask)
    if cover_map is None:
        print("No solutions (invalid open squares).")
        return

    open_mask = full_board_mask() & ~blocked_mask
    used = set()
    nodes = 0

    def dfs(remaining: int, depth: int) -> None:
        nonlocal nodes
        if nodes >= max_nodes:
            return
        if remaining == 0:
            print("  " * depth + "SOLUTION")
            nodes += 1
            return
        if depth >= depth_limit:
            print("  " * depth + "...")
            nodes += 1
            return

        cell = (remaining & -remaining).bit_length() - 1
        choices: List[Tuple[str, int]] = []
        for p in PIECE_NAMES:
            if p in used:
                continue
            for pm in cover_map[p][cell]:
                if (pm & remaining) != pm:
                    continue
                choices.append((p, pm))

        if not choices:
            print("  " * depth + "(dead end)")
            nodes += 1
            return

        for p, pm in choices[:max_children]:
            if nodes >= max_nodes:
                return
            print("  " * depth + f"{p}: {bitmask_to_coords(pm)} (mask {pm})")
            nodes += 1
            used.add(p)
            dfs(remaining ^ pm, depth + 1)
            used.remove(p)

        remaining_choices = len(choices) - max_children
        if remaining_choices > 0 and nodes < max_nodes:
            print("  " * depth + f"... {remaining_choices} more choices")
            nodes += 1


# =========================
# Main
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Given 7 blocker coordinates, show unique id, symmetry masks, parity, and solvability details."
    )
    parser.add_argument("coords", nargs="*", help="7 coords like: a1 b2 c3 d4 e5 f6 a2")
    parser.add_argument("--coords", dest="coords_str", default=None, help="coords as a single string")
    parser.add_argument(
        "--unique-csv",
        default=str(DATA_DIR / "01_unique_combos.csv"),
        help="unique combos CSV",
    )
    parser.add_argument("--max-solutions", type=int, default=0, help="0 = full count if needed")
    parser.add_argument(
        "--images-dir",
        default=str(OUTPUT_DIR / "images"),
        help="base directory for reflection images",
    )
    parser.add_argument(
        "--sample-solutions",
        type=int,
        default=0,
        help="print up to N sample solutions with per-piece placements",
    )
    parser.add_argument(
        "--skip-solve",
        action="store_true",
        help="skip solvability/solution count computation",
    )
    parser.add_argument(
        "--piece-breakdown",
        action="store_true",
        help="show per-piece placement frequency across sampled solutions",
    )
    parser.add_argument(
        "--breakdown-top",
        type=int,
        default=5,
        help="top placements per piece to show in breakdown",
    )
    parser.add_argument(
        "--breakdown-only",
        action="store_true",
        help="only output per-piece breakdown (suppresses other sections)",
    )
    parser.add_argument(
        "--solution-tree",
        action="store_true",
        help="print a partial DFS solution tree",
    )
    parser.add_argument(
        "--tree-depth",
        type=int,
        default=4,
        help="max depth of the solution tree",
    )
    parser.add_argument(
        "--tree-max-nodes",
        type=int,
        default=200,
        help="max nodes to print in the solution tree",
    )
    parser.add_argument(
        "--tree-max-children",
        type=int,
        default=50,
        help="max children per node to print",
    )
    parser.add_argument(
        "--tree-only",
        action="store_true",
        help="only output the solution tree",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="skip rendering reflection images",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        ensure_output_dir()
        coords = list(args.coords)
        if args.coords_str:
            coords.extend(args.coords_str.replace(",", " ").split())

        indices = parse_coords(coords)
        combo_mask = indices_to_bitmask(indices)
        canonical_mask = canonical_bitmask_from_indices(tuple(indices))

        unique_csv_path = Path(args.unique_csv)
        images_dir = Path(args.images_dir)

        unique_row = find_unique_row_by_canonical_bitmask(str(unique_csv_path), canonical_mask)
        if not unique_row:
            raise SystemExit("canonical bitmask not found in unique CSV")

        unique_id = unique_row["unique_id"]

        parity_possible, black_remaining, white_remaining, diff = classify_by_parity(canonical_mask)

        solution_count = None
        solvable = None

        if not args.breakdown_only and not args.tree_only:
            # Solvable / solution count
            if not args.skip_solve:
                max_solutions = args.max_solutions if args.max_solutions and args.max_solutions > 0 else None
                solution_count = count_solutions(canonical_mask, max_solutions=max_solutions)
                solvable = solution_count > 0

            sym_masks = symmetry_masks(indices)

            # Write reflection images
            if not args.skip_images:
                out_dir = images_dir / str(unique_id)
                reflections_dir = out_dir / "reflections"
                reflections_dir.mkdir(parents=True, exist_ok=True)
                for idx, mask in enumerate(sym_masks):
                    out_path = reflections_dir / f"reflection_{idx:02d}_{mask}.png"
                    render_bitmask_image(mask, str(out_path))

            print("=== INPUT ===")
            print(f"coords: {' '.join(coords)}")
            print(f"combo_bitmask: {combo_mask}")

            print("\n=== UNIQUE ===")
            print(f"unique_id: {unique_id}")
            print(f"canonical_bitmask: {unique_row['canonical_bitmask']}")
            print(f"canonical_coords: {unique_row['canonical_coords']}")

            print("\n=== SYMMETRY ===")
            print(f"reflection_bitmasks: {', '.join(str(m) for m in sym_masks)}")
            if not args.skip_images:
                print(f"reflection_images_dir: {reflections_dir}")

            print("\n=== PARITY ===")
            print(f"parity_possible: {1 if parity_possible else 0}")
            print(f"black_remaining: {black_remaining}")
            print(f"white_remaining: {white_remaining}")
            print(f"diff_black_minus_white: {diff}")

            print("\n=== SOLVABILITY ===")
            if args.skip_solve:
                print("solvable: (skipped)")
                print("solution_count: (skipped)")
            else:
                print(f"solvable: {1 if solvable else 0}")
                print(f"solution_count: {solution_count}")

        sample_count = max(args.sample_solutions, 0)
        if sample_count == 0 and not args.skip_solve and solution_count is not None:
            if solution_count <= 20:
                sample_count = solution_count
            else:
                try:
                    raw = input(
                        f"Found {solution_count} solutions. How many should be output? "
                    ).strip()
                    sample_count = max(int(raw), 0)
                except (ValueError, EOFError):
                    sample_count = 0
        if args.piece_breakdown and sample_count == 0:
            sample_count = 20
            if not args.breakdown_only:
                print("\n(note) --piece-breakdown enabled: defaulting --sample-solutions to 20")
        if args.breakdown_only:
            args.piece_breakdown = True
            sample_count = None

        if sample_count is None or sample_count > 0:
            per_piece_counts: Dict[str, Counter[int]] = defaultdict(Counter)
            if not args.breakdown_only:
                print("\n=== SAMPLE SOLUTIONS ===")
            solutions_dir = None
            if not args.skip_images:
                solutions_dir = images_dir / str(unique_id) / "solutions"
                solutions_dir.mkdir(parents=True, exist_ok=True)

            for idx, sol in enumerate(iter_solutions(canonical_mask, max_solutions=sample_count), start=1):
                if not args.breakdown_only:
                    print(f"solution {idx}:")
                for piece in PIECE_NAMES:
                    placement_mask = sol.get(piece)
                    if placement_mask is None:
                        continue
                    per_piece_counts[piece][placement_mask] += 1
                    if not args.breakdown_only:
                        print(f"  {piece}: {bitmask_to_coords(placement_mask)}")

                if solutions_dir is not None:
                    out_path = solutions_dir / f"solution_{idx:03d}.png"
                    render_solution_image(canonical_mask, sol, str(out_path))

            if args.piece_breakdown:
                print("=== PER-PIECE PLACEMENT BREAKDOWN (sampled) ===")
                for piece in PIECE_NAMES:
                    counter = per_piece_counts.get(piece, Counter())
                    total = sum(counter.values())
                    distinct = len(counter)
                    print(f"{piece}: sampled {total}, distinct placements {distinct}")
                    for placement_mask, count in counter.most_common(args.breakdown_top):
                        coords = bitmask_to_coords(placement_mask)
                        print(f"  {count}x {coords} (mask {placement_mask})")

        if args.solution_tree:
            header = "=== SOLUTION TREE (partial) ==="
            if args.breakdown_only or args.tree_only:
                print(header)
            else:
                print(f"\n{header}")
            print_solution_tree(
                canonical_mask,
                depth_limit=max(args.tree_depth, 0),
                max_nodes=max(args.tree_max_nodes, 1),
                max_children=max(args.tree_max_children, 1),
            )
    except Exception:
        LOGGER.exception("Failed to show board details")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
