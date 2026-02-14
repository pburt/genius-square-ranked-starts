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

from board_helpers import canonical_bitmask_from_indices

SCRIPT_PATH = SRC_DIR / "05_show_board_details.py"


def load_show_details_module():
    spec = importlib.util.spec_from_file_location("show_details", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load 05_show_board_details.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def coords_to_indices(coords):
    def coord_to_i(coord: str) -> int:
        coord = coord.strip().lower()
        row = ord(coord[0]) - ord("a")
        col = int(coord[1:]) - 1
        return row * 6 + col

    return [coord_to_i(c) for c in coords]


class ShowBoardDetailsTests(unittest.TestCase):
    def test_prompts_for_solution_limit_when_many(self) -> None:
        coords = ["a1", "a2", "a3", "a4", "a5", "a6", "b1"]
        indices = coords_to_indices(coords)
        canonical_mask = canonical_bitmask_from_indices(tuple(indices))

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            unique_csv = temp_path / "unique.csv"

            with unique_csv.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["unique_id", "canonical_bitmask", "orientations", "canonical_coords"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "unique_id": "1",
                        "canonical_bitmask": str(canonical_mask),
                        "orientations": "1",
                        "canonical_coords": " ".join(coords),
                    }
                )

            argv = [
                "05_show_board_details.py",
                *coords,
                "--unique-csv",
                str(unique_csv),
                "--sample-solutions",
                "0",
                "--skip-images",
            ]

            module = load_show_details_module()

            def fake_iter_solutions(mask, max_solutions=None):
                for _ in range(max_solutions or 0):
                    yield {"P1": 1}

            with unittest.mock.patch.object(sys, "argv", argv), unittest.mock.patch.object(
                module, "count_solutions", return_value=25
            ), unittest.mock.patch.object(
                module, "iter_solutions", side_effect=fake_iter_solutions
            ) as iter_mock, unittest.mock.patch(
                "builtins.input", return_value="3"
            ):
                module.main()

            self.assertTrue(iter_mock.called)
            args, kwargs = iter_mock.call_args
            self.assertEqual(kwargs.get("max_solutions"), 3)


if __name__ == "__main__":
    unittest.main()
