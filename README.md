# Genius Square: Ranked Starts

Genius Square without the game-dice restriction has 8,347,680 possible 7‑blocker starts on a 6×6 board. We collapse those by D4 symmetry into 1,044,690 unique starting positions, filter by parity (1,036,620 parity‑possible; 8,070 parity‑impossible), and then solve the remaining boards to determine which starts are actually solvable (1,022,933 solvable; 13,687 unsolvable) and how many solutions each has.

Utilities for generating, filtering, and analyzing 7-blocker starting boards on a 6×6 grid under D4 symmetries. The scripts compute unique blocker combinations, parity feasibility, solvability, and detailed board diagnostics.

For a fun challenge, try the starts with exactly 1 solution, or pick a start with only a handful of solutions and see if you can find them all in [data/04_solvable_with_solution_counts.csv](data/04_solvable_with_solution_counts.csv).

## TL;DR

Run `python src/05_show_board_details.py a1 b2 c3 d4 e5 f6 a2` to produce solutions with images for a specific start.

## Requirements

- Python 3.10+
- Dependencies listed in requirements.txt

Install deps:

- Create a virtual environment (optional)
- Install: `pip install -r requirements.txt`

## Project layout

- src/board_helpers.py — core board math, symmetry, parity checks, polyomino placements, solver
- src/01_unique_combos.py — generate unique blocker combos and symmetry mapping
- src/02_parity_filter.py — split unique combos by parity feasibility
- src/03_solve_progress.py — solve parity-feasible boards and split solvable vs unsolvable
- src/04_recount_solution_counts.py — recompute full solution counts in parallel and update CSV
- src/05_show_board_details.py — inspect a specific board, render reflections, sample solutions

CSV source lists live under data/. Scripts write generated outputs to output/ by default.

Note: data/01_sym_to_unique_id.csv is ignored in git due to GitHub size limits. Regenerate it with step 1 in the pipeline.

CSV outputs are written to the working directory by default.

## Typical pipeline

1) Generate unique combos and symmetry map

`python src/01_unique_combos.py --unique-out output/01_unique_combos.csv --map-out output/01_sym_to_unique_id.csv --workers 0`

For maximum throughput, add --unordered. If you need sorted CSVs, add --sort-after (memory heavy for large files).

2) Parity filter

`python src/02_parity_filter.py --input data/01_unique_combos.csv --possible-out output/02_parity_solvable.csv --impossible-out output/02_parity_unsolvable.csv --workers 0`

3) Solve parity-possible boards

`python src/03_solve_progress.py --input data/02_parity_solvable.csv --solvable-out output/03_solvable.csv --unsolvable-out output/03_unsolvable.csv --workers 0`

4) Recount full solution counts (optional, parallel)

`python src/04_recount_solution_counts.py --input data/03_solvable.csv --output output/04_solvable_with_solution_counts.csv --workers 0 --chunksize 100 --resume`

To chain from generated outputs instead of the static data files, add --use-output-inputs to steps 2–4.

## Inspect a specific board

Provide 7 coordinates like `a1 b2 c3 d4 e5 f6 a2`:

`python src/05_show_board_details.py a1 b2 c3 d4 e5 f6 a2 --unique-csv data/01_unique_combos.csv --sample-solutions 5 --piece-breakdown`

This prints parity and solvability details, optionally renders reflection images into a directory named after the unique ID.

## Notes

- The full combination space is large; generating unique combos is CPU-intensive.
- The solver is exact and can be slow; consider using `--max-solutions` to short-circuit.

## Column reference

- unique_id: Integer ID for a canonical blocker configuration.
- canonical_bitmask: Bitmask of the canonical blocker positions under D4 symmetry.
- combo_bitmask: Bitmask for a specific blocker combo (non-canonical).
- orientations: Count of distinct D4 orientations for the canonical mask.
- canonical_coords: Space-separated coordinates for the canonical blocker set.
- parity_possible: 1 if parity constraints allow a solution, else 0.
- black_remaining / white_remaining: Remaining square counts after blockers.
- diff_black_minus_white: black_remaining - white_remaining.
- solution_count: Number of tilings/solutions found for a board.
