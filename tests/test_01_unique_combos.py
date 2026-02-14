import csv
import sys
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SCRIPT_PATH = SRC_DIR / "01_unique_combos.py"


def load_unique_combos_module():
    spec = importlib.util.spec_from_file_location("unique_combos", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load 01_unique_combos.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UniqueCombosScriptTests(unittest.TestCase):
    def test_writes_expected_csv_headers_and_rows(self) -> None:
        combos = [tuple(range(7)), tuple(range(1, 8))]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            unique_out = temp_path / "unique.csv"
            map_out = temp_path / "map.csv"

            argv = [
                "01_unique_combos.py",
                "--unique-out",
                str(unique_out),
                "--map-out",
                str(map_out),
                "--workers",
                "1",
                "--progress-every",
                "1000000",
            ]

            unique_combos = load_unique_combos_module()
            with patch.object(sys, "argv", argv), patch.object(
                unique_combos, "combinations", return_value=iter(combos)
            ):
                unique_combos.main()

            with unique_out.open(newline="") as unique_file:
                rows = list(csv.reader(unique_file))

            with map_out.open(newline="") as map_file:
                map_rows = list(csv.reader(map_file))

        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(
            rows[0],
            ["unique_id", "canonical_bitmask", "orientations", "canonical_coords"],
        )
        self.assertEqual(map_rows[0], ["combo_bitmask", "unique_id"])
        self.assertEqual(len(map_rows) - 1, len(combos))


if __name__ == "__main__":
    unittest.main()
