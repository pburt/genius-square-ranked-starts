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

SCRIPT_PATH = SRC_DIR / "04_recount_solution_counts.py"


def load_recount_module():
    spec = importlib.util.spec_from_file_location("recount", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load 04_recount_solution_counts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecountSolutionCountsTests(unittest.TestCase):
    def test_recount_sorts_by_solution_count(self) -> None:
        rows = [
            {
                "unique_id": "0",
                "canonical_bitmask": "1",
                "orientations": "1",
                "canonical_coords": "a1 a2 a3 a4 a5 a6 b1",
            },
            {
                "unique_id": "1",
                "canonical_bitmask": "2",
                "orientations": "1",
                "canonical_coords": "b2 b3 b4 b5 b6 c1 c2",
            },
            {
                "unique_id": "2",
                "canonical_bitmask": "3",
                "orientations": "1",
                "canonical_coords": "c1 c2 c3 c4 c5 c6 d1",
            },
        ]
        count_map = {1: 5, 2: 1, 3: 3}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.csv"
            output_path = temp_path / "output.csv"

            with input_path.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["unique_id", "canonical_bitmask", "orientations", "canonical_coords"],
                )
                writer.writeheader()
                writer.writerows(rows)

            argv = [
                "04_recount_solution_counts.py",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--workers",
                "1",
                "--progress-every",
                "1000000",
                "--fsync-every",
                "0",
            ]

            recount = load_recount_module()

            def fake_count(mask: int) -> int:
                return count_map[mask]

            with unittest.mock.patch.object(sys, "argv", argv), unittest.mock.patch.object(
                recount, "_count_for_mask", side_effect=fake_count
            ):
                recount.main()

            with output_path.open(newline="") as file:
                output_rows = list(csv.DictReader(file))

        solution_counts = [int(row["solution_count"]) for row in output_rows]
        self.assertEqual(solution_counts, sorted(solution_counts))


if __name__ == "__main__":
    unittest.main()
