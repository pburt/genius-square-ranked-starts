import csv
import importlib.util
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from board_helpers import classify_by_parity

SCRIPT_PATH = SRC_DIR / "02_parity_filter.py"


def load_parity_filter_module():
    spec = importlib.util.spec_from_file_location("parity_filter", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load 02_parity_filter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ParityFilterScriptTests(unittest.TestCase):
    def test_splits_rows_by_parity(self) -> None:
        rows = [
            {
                "unique_id": "0",
                "canonical_bitmask": str(2**0 + 2**1 + 2**2 + 2**3 + 2**4 + 2**5 + 2**6),
                "orientations": "1",
                "canonical_coords": "a1 a2 a3 a4 a5 a6 b1",
            },
            {
                "unique_id": "1",
                "canonical_bitmask": str(2**7 + 2**8 + 2**9 + 2**10 + 2**11 + 2**12 + 2**13),
                "orientations": "1",
                "canonical_coords": "b2 b3 b4 b5 b6 c1 c2",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.csv"
            possible_path = temp_path / "possible.csv"
            impossible_path = temp_path / "impossible.csv"

            with input_path.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["unique_id", "canonical_bitmask", "orientations", "canonical_coords"],
                )
                writer.writeheader()
                writer.writerows(rows)

            argv = [
                "02_parity_filter.py",
                "--input",
                str(input_path),
                "--possible-out",
                str(possible_path),
                "--impossible-out",
                str(impossible_path),
                "--progress-every",
                "1000000",
            ]

            parity_filter = load_parity_filter_module()
            with unittest.mock.patch.object(sys, "argv", argv):
                parity_filter.main()

            with possible_path.open(newline="") as file:
                possible_rows = list(csv.DictReader(file))

            with impossible_path.open(newline="") as file:
                impossible_rows = list(csv.DictReader(file))

        expected_possible = []
        expected_impossible = []
        for row in rows:
            mask = int(row["canonical_bitmask"])
            parity_possible, _, _, _ = classify_by_parity(mask)
            if parity_possible:
                expected_possible.append(row["unique_id"])
            else:
                expected_impossible.append(row["unique_id"])

        self.assertEqual(
            sorted(r["unique_id"] for r in possible_rows),
            sorted(expected_possible),
        )
        self.assertEqual(
            sorted(r["unique_id"] for r in impossible_rows),
            sorted(expected_impossible),
        )


if __name__ == "__main__":
    unittest.main()
