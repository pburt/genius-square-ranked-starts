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

SCRIPT_PATH = SRC_DIR / "03_solve_progress.py"


def load_solve_progress_module():
    spec = importlib.util.spec_from_file_location("solve_progress", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load 03_solve_progress.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SolveProgressScriptTests(unittest.TestCase):
    def test_splits_solvable_and_unsolvable(self) -> None:
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
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.csv"
            solvable_path = temp_path / "solvable.csv"
            unsolvable_path = temp_path / "unsolvable.csv"

            with input_path.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["unique_id", "canonical_bitmask", "orientations", "canonical_coords"],
                )
                writer.writeheader()
                writer.writerows(rows)

            argv = [
                "03_solve_progress.py",
                "--input",
                str(input_path),
                "--solvable-out",
                str(solvable_path),
                "--unsolvable-out",
                str(unsolvable_path),
                "--workers",
                "1",
                "--progress-every",
                "1000000",
            ]

            solve_progress = load_solve_progress_module()

            def fake_count_solutions(mask: int, max_solutions=None):
                return 2 if mask % 2 == 0 else 0

            with unittest.mock.patch.object(sys, "argv", argv), unittest.mock.patch.object(
                solve_progress, "count_solutions", side_effect=fake_count_solutions
            ):
                solve_progress.main()

            with solvable_path.open(newline="") as file:
                solvable_rows = list(csv.DictReader(file))

            with unsolvable_path.open(newline="") as file:
                unsolvable_rows = list(csv.DictReader(file))

        self.assertEqual([row["unique_id"] for row in solvable_rows], ["1"])
        self.assertEqual([row["unique_id"] for row in unsolvable_rows], ["0"])
        self.assertEqual(solvable_rows[0]["solution_count"], "2")
        self.assertEqual(unsolvable_rows[0]["solution_count"], "0")


if __name__ == "__main__":
    unittest.main()
