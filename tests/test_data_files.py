import csv
import random
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from board_helpers import (
    canonical_bitmask_from_indices,
    classify_by_parity,
    count_solutions,
)

SAMPLE_SIZE = 10
RANDOM_SEED = 1337

REQUIRED_COLUMNS = {
    "01_unique_combos.csv": [
        "unique_id",
        "canonical_bitmask",
        "orientations",
        "canonical_coords",
    ],
    "01_sym_to_unique_id.csv": ["combo_bitmask", "unique_id"],
    "02_parity_solvable.csv": [
        "unique_id",
        "canonical_bitmask",
        "orientations",
        "canonical_coords",
        "black_remaining",
        "white_remaining",
        "diff_black_minus_white",
        "parity_possible",
    ],
    "02_parity_unsolvable.csv": [
        "unique_id",
        "canonical_bitmask",
        "orientations",
        "canonical_coords",
        "black_remaining",
        "white_remaining",
        "diff_black_minus_white",
        "parity_possible",
    ],
    "03_solvable.csv": [
        "unique_id",
        "canonical_bitmask",
        "orientations",
        "canonical_coords",
        "solution_count",
    ],
    "03_unsolvable.csv": [
        "unique_id",
        "canonical_bitmask",
        "orientations",
        "canonical_coords",
        "solution_count",
    ],
    "04_solvable_with_solution_counts.csv": [
        "unique_id",
        "canonical_bitmask",
        "orientations",
        "canonical_coords",
        "solution_count",
    ],
}

INTEGER_COLUMNS = {
    "01_unique_combos.csv": ["unique_id", "canonical_bitmask", "orientations"],
    "01_sym_to_unique_id.csv": ["combo_bitmask", "unique_id"],
    "02_parity_solvable.csv": [
        "unique_id",
        "canonical_bitmask",
        "orientations",
        "black_remaining",
        "white_remaining",
        "diff_black_minus_white",
        "parity_possible",
    ],
    "02_parity_unsolvable.csv": [
        "unique_id",
        "canonical_bitmask",
        "orientations",
        "black_remaining",
        "white_remaining",
        "diff_black_minus_white",
        "parity_possible",
    ],
    "03_solvable.csv": [
        "unique_id",
        "canonical_bitmask",
        "orientations",
        "solution_count",
    ],
    "03_unsolvable.csv": [
        "unique_id",
        "canonical_bitmask",
        "orientations",
        "solution_count",
    ],
    "04_solvable_with_solution_counts.csv": [
        "unique_id",
        "canonical_bitmask",
        "orientations",
        "solution_count",
    ],
}


def reservoir_sample(rows, k, rng):
    sample = []
    for i, row in enumerate(rows):
        if i < k:
            sample.append(row)
        else:
            j = rng.randint(0, i)
            if j < k:
                sample[j] = row
    return sample


def coord_to_index(coord: str) -> int:
    coord = coord.strip().lower()
    row = ord(coord[0]) - ord("a")
    col = int(coord[1:]) - 1
    return row * 6 + col


def coords_to_indices(coords: str):
    return [coord_to_index(c) for c in coords.split() if c]


def indices_to_bitmask(indices) -> int:
    mask = 0
    for i in indices:
        mask |= 1 << i
    return mask


class DataFilesSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        unique_path = DATA_DIR / "01_unique_combos.csv"
        cls.unique_id_to_canonical = {}
        if unique_path.exists():
            with unique_path.open("r", newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    cls.unique_id_to_canonical[row["unique_id"]] = int(row["canonical_bitmask"])

    def test_data_files_exist(self) -> None:
        for filename in REQUIRED_COLUMNS:
            path = DATA_DIR / filename
            print(f"[data-check] exists: {path}")
            self.assertTrue(path.exists(), f"missing {path}")

    def test_sample_rows_match_format(self) -> None:
        rng = random.Random(RANDOM_SEED)
        for filename, required_cols in REQUIRED_COLUMNS.items():
            path = DATA_DIR / filename
            print(f"[data-check] reading: {path}")
            with path.open("r", newline="") as file:
                reader = csv.DictReader(file)
                fieldnames = reader.fieldnames or []

                for col in required_cols:
                    self.assertIn(col, fieldnames, f"{filename} missing column {col}")

                rows = reservoir_sample(reader, SAMPLE_SIZE, rng)

            print(f"[data-check] sampled {len(rows)} rows from {filename}")

            for row in rows:
                uid = row.get("unique_id", "")
                print(f"[data-check] validating row unique_id={uid}")
                for col in required_cols:
                    self.assertIn(col, row)
                    self.assertNotEqual(row[col], "")

                for col in INTEGER_COLUMNS.get(filename, []):
                    try:
                        int(row[col])
                    except ValueError as exc:
                        raise AssertionError(
                            f"{filename} invalid int in column {col}: {row[col]}"
                        ) from exc

                # Validate canonical_coords <-> canonical_bitmask consistency when present
                if "canonical_coords" in row and "canonical_bitmask" in row:
                    print("[data-check]  canonical coords/bitmask consistency")
                    indices = coords_to_indices(row["canonical_coords"])
                    self.assertEqual(len(indices), 7, f"{filename} expected 7 coords")
                    mask_from_coords = indices_to_bitmask(indices)
                    canonical_from_coords = canonical_bitmask_from_indices(tuple(indices))
                    self.assertEqual(
                        int(row["canonical_bitmask"]),
                        canonical_from_coords,
                        f"{filename} canonical_bitmask mismatch",
                    )
                    # Ensure coords are canonical representation for that bitmask
                    self.assertEqual(mask_from_coords, canonical_from_coords)

                # Validate unique_id -> canonical_bitmask consistency when possible
                if "unique_id" in row and "canonical_bitmask" in row:
                    print("[data-check]  unique_id -> canonical_bitmask mapping")
                    expected = self.unique_id_to_canonical.get(row["unique_id"])
                    if expected is not None:
                        self.assertEqual(
                            int(row["canonical_bitmask"]),
                            expected,
                            f"{filename} unique_id mapping mismatch",
                        )

                # Validate parity fields
                if "parity_possible" in row:
                    print("[data-check]  parity recompute")
                    parity_possible, black_remaining, white_remaining, diff = classify_by_parity(
                        int(row["canonical_bitmask"])
                    )
                    self.assertEqual(int(row["parity_possible"]), 1 if parity_possible else 0)
                    self.assertEqual(int(row["black_remaining"]), black_remaining)
                    self.assertEqual(int(row["white_remaining"]), white_remaining)
                    self.assertEqual(int(row["diff_black_minus_white"]), diff)

                # Validate solution_count expectations and recompute counts (deep check)
                if "solution_count" in row:
                    print("[data-check]  solution_count recompute")
                    if filename in {"03_solvable.csv", "03_unsolvable.csv"}:
                        recomputed = count_solutions(
                            int(row["canonical_bitmask"]),
                            max_solutions=1,
                        )
                    else:
                        recomputed = count_solutions(
                            int(row["canonical_bitmask"]),
                            max_solutions=None,
                        )
                    self.assertEqual(int(row["solution_count"]), recomputed)

                if filename in {"03_solvable.csv", "04_solvable_with_solution_counts.csv"}:
                    self.assertGreater(int(row["solution_count"]), 0)
                if filename == "03_unsolvable.csv":
                    self.assertEqual(int(row["solution_count"]), 0)


if __name__ == "__main__":
    unittest.main()
