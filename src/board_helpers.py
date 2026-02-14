"""Core board helpers for the 6x6 blocker/tiling problem."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

# =========================
# Board configuration
# =========================
N = 6
NUM_SQUARES = N * N  # 36
K_BLOCKERS = 7
TOTAL_BLACK = 18
TOTAL_WHITE = 18

LOGGER = logging.getLogger(__name__)

# =========================
# Index / coord helpers
# =========================
def index_to_row_col(index: int) -> Tuple[int, int]:
    """Convert a linear index into (row, col)."""
    return divmod(index, N)


def row_col_to_index(row: int, col: int) -> int:
    """Convert (row, col) into a linear index."""
    return row * N + col


def index_to_coord(index: int) -> str:
    """Convert a linear index into a board coordinate like 'a1'."""
    row, col = index_to_row_col(index)
    return f"{chr(ord('a') + row)}{col + 1}"


def indices_to_coords(indices: Tuple[int, ...]) -> str:
    """Convert a tuple of indices to space-separated coordinates."""
    return " ".join(index_to_coord(index) for index in indices)

# =========================
# Bit helpers
# =========================
def count_set_bits(value: int) -> int:
    """Return the number of set bits in value."""
    return bin(value).count("1")


def iter_set_bits(mask: int) -> Iterable[int]:
    """Yield set bit positions from a bitmask."""
    while mask:
        least_significant_bit = mask & -mask
        yield least_significant_bit.bit_length() - 1
        mask ^= least_significant_bit


def full_board_mask() -> int:
    """Return a bitmask with all board squares set."""
    return (1 << NUM_SQUARES) - 1


def indices_to_bitmask(indices: Iterable[int]) -> int:
    """Convert an iterable of indices into a bitmask."""
    mask = 0
    for index in indices:
        mask |= 1 << index
    return mask


def bitmask_to_indices(mask: int) -> Tuple[int, ...]:
    """Convert a bitmask into a tuple of indices."""
    return tuple(index for index in range(NUM_SQUARES) if (mask >> index) & 1)


def bitmask_to_coords(mask: int) -> str:
    """Convert a bitmask into space-separated coordinates."""
    return indices_to_coords(bitmask_to_indices(mask))

# =========================
# D4 symmetries (8 transforms)
# =========================
@lru_cache(maxsize=1)
def build_symmetry_maps() -> List[List[int]]:
    """Build D4 symmetry maps as index-to-index transforms."""

    def rot90(row: int, col: int) -> Tuple[int, int]:
        return (col, N - 1 - row)

    def reflect(row: int, col: int) -> Tuple[int, int]:
        return (row, N - 1 - col)

    def apply_rot(row: int, col: int, turns: int) -> Tuple[int, int]:
        for _ in range(turns % 4):
            row, col = rot90(row, col)
        return row, col

    symmetry_maps: List[List[int]] = []

    # rotations
    for turns in range(4):
        mapping = [0] * NUM_SQUARES
        for index in range(NUM_SQUARES):
            row, col = index_to_row_col(index)
            new_row, new_col = apply_rot(row, col, turns)
            mapping[index] = row_col_to_index(new_row, new_col)
        symmetry_maps.append(mapping)

    # reflections + rotations
    for turns in range(4):
        mapping = [0] * NUM_SQUARES
        for index in range(NUM_SQUARES):
            row, col = index_to_row_col(index)
            new_row, new_col = apply_rot(row, col, turns)
            new_row, new_col = reflect(new_row, new_col)
            mapping[index] = row_col_to_index(new_row, new_col)
        symmetry_maps.append(mapping)

    return symmetry_maps


SYM_MAPS = build_symmetry_maps()


def canonical_bitmask_from_indices(indices: Tuple[int, ...]) -> int:
    """Return the canonical bitmask for indices under D4 symmetries."""
    best_mask: Optional[int] = None
    for symmetry_map in SYM_MAPS:
        mask = 0
        for index in indices:
            mask |= 1 << symmetry_map[index]
        if best_mask is None or mask < best_mask:
            best_mask = mask
    return best_mask  # type: ignore[return-value]

# =========================
# Parity helpers
# =========================
def is_black_square(index: int) -> bool:
    """Return True if the index is on a black square in a checkerboard pattern."""
    row, col = index_to_row_col(index)
    return (row + col) % 2 == 0


COLOR = [1 if is_black_square(i) else 0 for i in range(NUM_SQUARES)]  # 1=black, 0=white


def classify_by_parity(blocker_bitmask: int) -> Tuple[bool, int, int, int]:
    """
    Return (parity_possible, black_remaining, white_remaining, diff).

    diff = black_remaining - white_remaining
    """
    black_blockers = 0
    for index in iter_set_bits(blocker_bitmask):
        black_blockers += COLOR[index]

    white_blockers = K_BLOCKERS - black_blockers

    black_remaining = TOTAL_BLACK - black_blockers
    white_remaining = TOTAL_WHITE - white_blockers
    diff = black_remaining - white_remaining

    parity_possible = (abs(diff) <= 5) and (diff % 2 != 0)
    return parity_possible, black_remaining, white_remaining, diff

# =========================
# Polyomino definitions
# =========================
PIECES_BASE: Dict[str, List[Tuple[int, int]]] = {
    "P1":  [(0, 0)],
    "P2":  [(0, 0), (0, 1)],
    "P3I": [(0, 0), (0, 1), (0, 2)],
    "P3L": [(0, 0), (1, 0), (1, 1)],
    "P4I": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "P4O": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "P4T": [(0, 0), (0, 1), (0, 2), (1, 1)],
    "P4L": [(0, 0), (1, 0), (2, 0), (2, 1)],
    "P4S": [(0, 1), (0, 2), (1, 0), (1, 1)],
}

PIECE_NAMES = list(PIECES_BASE.keys())


def rotate_cells_90(cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Rotate cells 90 degrees around the origin."""
    return [(-col, row) for row, col in cells]


def reflect_cells_x(cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Reflect cells across the vertical axis."""
    return [(row, -col) for row, col in cells]


def normalize_cells(cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Normalize cells so that the minimum row/col is at the origin."""
    min_row = min(row for row, _ in cells)
    min_col = min(col for _, col in cells)
    return sorted((row - min_row, col - min_col) for row, col in cells)


def get_unique_orientations(base: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """Return all distinct orientations (rotations/reflections) of a polyomino."""
    seen = set()
    orientations: List[List[Tuple[int, int]]] = []
    current = base
    for _ in range(4):
        for variant in (current, reflect_cells_x(current)):
            normalized = tuple(normalize_cells(variant))
            if normalized not in seen:
                seen.add(normalized)
                orientations.append(list(normalized))
        current = rotate_cells_90(current)
    return orientations


@lru_cache(maxsize=1)
def precompute_piece_placements() -> Dict[str, List[int]]:
    """Precompute all legal placements of each polyomino on the board."""
    placements: Dict[str, List[int]] = {piece: [] for piece in PIECE_NAMES}

    for piece, base in PIECES_BASE.items():
        for orientation in get_unique_orientations(base):
            max_row = max(row for row, _ in orientation)
            max_col = max(col for _, col in orientation)
            for base_row in range(N - max_row):
                for base_col in range(N - max_col):
                    mask = 0
                    for row, col in orientation:
                        mask |= 1 << row_col_to_index(base_row + row, base_col + col)
                    placements[piece].append(mask)

        placements[piece] = sorted(set(placements[piece]))

    return placements


ALL_PLACEMENTS = precompute_piece_placements()

# =========================
# Solver (count only)
# =========================
def build_cover_map(blocked_mask: int) -> Optional[Dict[str, List[List[int]]]]:
    """Build per-piece, per-cell placement lists for the given blockers."""
    if blocked_mask & ~full_board_mask():
        LOGGER.warning("Blocked mask has bits outside the board; returning no cover map.")
        return None

    open_mask = full_board_mask() & ~blocked_mask
    if count_set_bits(open_mask) != 29:
        return None

    valid_by_piece = {
        piece: [mask for mask in ALL_PLACEMENTS[piece] if (mask & blocked_mask) == 0]
        for piece in PIECE_NAMES
    }

    cover_map: Dict[str, List[List[int]]] = {}
    for piece in PIECE_NAMES:
        per_cell = [[] for _ in range(NUM_SQUARES)]
        for placement_mask in valid_by_piece[piece]:
            for index in iter_set_bits(placement_mask):
                per_cell[index].append(placement_mask)
        cover_map[piece] = per_cell

    return cover_map


def count_solutions(blocked_mask: int, max_solutions: Optional[int] = None) -> int:
    """Count solutions for a blocker mask. Optionally stop after max_solutions."""
    open_mask = full_board_mask() & ~blocked_mask
    cover_map = build_cover_map(blocked_mask)
    if cover_map is None:
        return 0

    used_pieces = set()
    solutions = 0

    def dfs(remaining: int) -> None:
        nonlocal solutions
        if remaining == 0:
            solutions += 1
            return
        if max_solutions and solutions >= max_solutions:
            return

        cell = (remaining & -remaining).bit_length() - 1
        for piece in PIECE_NAMES:
            if piece in used_pieces:
                continue
            for placement_mask in cover_map[piece][cell]:
                if (placement_mask & remaining) != placement_mask:
                    continue
                used_pieces.add(piece)
                dfs(remaining ^ placement_mask)
                used_pieces.remove(piece)
                if max_solutions and solutions >= max_solutions:
                    return

    dfs(open_mask)
    return solutions


def iter_solutions(
    blocked_mask: int, max_solutions: Optional[int] = None
) -> Iterator[Dict[str, int]]:
    """Yield solution placements for a blocker mask."""
    open_mask = full_board_mask() & ~blocked_mask
    cover_map = build_cover_map(blocked_mask)
    if cover_map is None:
        return

    used_pieces = set()
    placement_by_piece: Dict[str, int] = {}
    solutions = 0

    def dfs(remaining: int) -> Iterator[Dict[str, int]]:
        nonlocal solutions
        if remaining == 0:
            solutions += 1
            yield dict(placement_by_piece)
            return
        if max_solutions and solutions >= max_solutions:
            return

        cell = (remaining & -remaining).bit_length() - 1
        for piece in PIECE_NAMES:
            if piece in used_pieces:
                continue
            for placement_mask in cover_map[piece][cell]:
                if (placement_mask & remaining) != placement_mask:
                    continue
                used_pieces.add(piece)
                placement_by_piece[piece] = placement_mask
                yield from dfs(remaining ^ placement_mask)
                used_pieces.remove(piece)
                placement_by_piece.pop(piece, None)
                if max_solutions and solutions >= max_solutions:
                    return

    yield from dfs(open_mask)
