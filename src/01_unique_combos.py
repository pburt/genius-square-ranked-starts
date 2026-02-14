import argparse
import csv
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Dict

from board_helpers import (
    K_BLOCKERS,
    NUM_SQUARES,
    SYM_MAPS,
    canonical_bitmask_from_indices,
    indices_to_bitmask,
    bitmask_to_indices,
    indices_to_coords,
)
from path_helpers import DATA_DIR, OUTPUT_DIR, ensure_output_dir

TOTAL_COMBOS = 8_347_680  # comb(36,7)

LOGGER = logging.getLogger(__name__)

# Precompute bit values for each symmetry map to avoid repeated shifts in the inner loop.
SYM_BIT_MAPS = [[1 << mapped_index for mapped_index in symmetry_map] for symmetry_map in SYM_MAPS]


def compute_combo_masks(combo: tuple[int, ...]) -> tuple[int, int]:
    """Return (combo_mask, canonical_mask) for a blocker combo."""
    combo_mask = indices_to_bitmask(combo)
    canonical_mask = canonical_bitmask_from_indices(combo)
    return combo_mask, canonical_mask


def sort_csv_by_column(file_path: Path, column: str) -> None:
    """Sort a CSV file by a numeric column in-place (memory heavy for large files)."""
    with file_path.open("r", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    rows.sort(key=lambda row: int(row[column]))

    with file_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate unique blocker combos up to symmetry and a mapping from all combos."
    )
    parser.add_argument(
        "--unique-out",
        default=str(OUTPUT_DIR / "01_unique_combos.csv"),
        help="unique combos CSV output",
    )
    parser.add_argument(
        "--map-out",
        default=str(OUTPUT_DIR / "01_sym_to_unique_id.csv"),
        help="symmetry map CSV output",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="number of worker processes (0 = cpu count, 1 = no multiprocessing)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1000,
        help="task chunksize for multiprocessing",
    )
    parser.add_argument(
        "--unordered",
        action="store_true",
        help="process combos out of order for higher throughput",
    )
    parser.add_argument(
        "--sort-after",
        action="store_true",
        help="sort output CSVs at the end (memory heavy for large files)",
    )
    parser.add_argument("--progress-every", type=int, default=100_000, help="progress interval")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        ensure_output_dir()
        unique_out_path = Path(args.unique_out)
        map_out_path = Path(args.map_out)

        canonical_mask_to_id: Dict[int, int] = {}

        all_indices = range(NUM_SQUARES)
        next_unique_id = 0
        processed = 0
        start_time = time.time()
        ema_rate: float | None = None
        ema_alpha = 0.2

        sym_bit_maps = SYM_BIT_MAPS
        worker_count = args.workers if args.workers >= 0 else 0
        if worker_count == 0:
            worker_count = os.cpu_count() or 1
        use_multiprocessing = worker_count > 1
        use_unordered = args.unordered and use_multiprocessing

        with (
            open(unique_out_path, "w", newline="") as unique_file,
            open(map_out_path, "w", newline="") as map_file,
        ):
            unique_writer = csv.writer(unique_file)
            map_writer = csv.writer(map_file)

            # Write CSV headers
            unique_writer.writerow(
                [
                    "unique_id",
                    "canonical_bitmask",
                    "orientations",
                    "canonical_coords",
                ]
            )
            map_writer.writerow(["combo_bitmask", "unique_id"])

            combo_iter = combinations(all_indices, K_BLOCKERS)

            if use_multiprocessing:
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    if use_unordered:
                        futures = {}
                        in_flight_limit = max(1, worker_count * 4)

                        def submit_next() -> None:
                            try:
                                combo = next(combo_iter)
                            except StopIteration:
                                return
                            future = executor.submit(compute_combo_masks, combo)
                            futures[future] = None

                        for _ in range(in_flight_limit):
                            submit_next()

                        while futures:
                            for future in as_completed(futures):
                                futures.pop(future, None)
                                combo_mask, canonical_mask = future.result()
                                processed += 1

                                unique_id = canonical_mask_to_id.get(canonical_mask)

                                if unique_id is None:
                                    unique_id = next_unique_id
                                    canonical_mask_to_id[canonical_mask] = unique_id
                                    next_unique_id += 1

                                    canonical_indices = bitmask_to_indices(canonical_mask)

                                    # Count distinct orientations under D4 symmetries
                                    orientation_masks = set()
                                    for symmetry_bit_map in sym_bit_maps:
                                        mask = 0
                                        for index in canonical_indices:
                                            mask |= symmetry_bit_map[index]
                                        orientation_masks.add(mask)

                                    unique_writer.writerow(
                                        [
                                            unique_id,
                                            canonical_mask,
                                            len(orientation_masks),
                                            indices_to_coords(canonical_indices),
                                        ]
                                    )

                                map_writer.writerow([combo_mask, unique_id])

                                if processed % args.progress_every == 0:
                                    elapsed = time.time() - start_time
                                    rate = processed / elapsed if elapsed > 0 else 0.0
                                    ema_rate = rate if ema_rate is None else (ema_alpha * rate + (1 - ema_alpha) * ema_rate)
                                    remaining = TOTAL_COMBOS - processed
                                    eta = remaining / ema_rate if ema_rate and ema_rate > 0 else 0

                                    LOGGER.info(
                                        "[%s/%s] %.2f%% | %.0f combos/s | ETA %.1f min",
                                        f"{processed:,}",
                                        f"{TOTAL_COMBOS:,}",
                                        (processed / TOTAL_COMBOS) * 100,
                                        rate,
                                        eta / 60,
                                    )

                                submit_next()
                                break
                    else:
                        results = executor.map(
                            compute_combo_masks,
                            combo_iter,
                            chunksize=max(args.chunksize, 1),
                        )
                        for combo_mask, canonical_mask in results:
                            processed += 1

                            unique_id = canonical_mask_to_id.get(canonical_mask)

                            if unique_id is None:
                                unique_id = next_unique_id
                                canonical_mask_to_id[canonical_mask] = unique_id
                                next_unique_id += 1

                                canonical_indices = bitmask_to_indices(canonical_mask)

                                # Count distinct orientations under D4 symmetries
                                orientation_masks = set()
                                for symmetry_bit_map in sym_bit_maps:
                                    mask = 0
                                    for index in canonical_indices:
                                        mask |= symmetry_bit_map[index]
                                    orientation_masks.add(mask)

                                unique_writer.writerow(
                                    [
                                        unique_id,
                                        canonical_mask,
                                        len(orientation_masks),
                                        indices_to_coords(canonical_indices),
                                    ]
                                )

                            map_writer.writerow([combo_mask, unique_id])

                            if processed % args.progress_every == 0:
                                elapsed = time.time() - start_time
                                rate = processed / elapsed if elapsed > 0 else 0.0
                                ema_rate = rate if ema_rate is None else (ema_alpha * rate + (1 - ema_alpha) * ema_rate)
                                remaining = TOTAL_COMBOS - processed
                                eta = remaining / ema_rate if ema_rate and ema_rate > 0 else 0

                                LOGGER.info(
                                    "[%s/%s] %.2f%% | %.0f combos/s | ETA %.1f min",
                                    f"{processed:,}",
                                    f"{TOTAL_COMBOS:,}",
                                    (processed / TOTAL_COMBOS) * 100,
                                    rate,
                                    eta / 60,
                                )
            else:
                for combo in combo_iter:
                    processed += 1

                    combo_mask, canonical_mask = compute_combo_masks(combo)
                    unique_id = canonical_mask_to_id.get(canonical_mask)

                    if unique_id is None:
                        unique_id = next_unique_id
                        canonical_mask_to_id[canonical_mask] = unique_id
                        next_unique_id += 1

                        canonical_indices = bitmask_to_indices(canonical_mask)

                        # Count distinct orientations under D4 symmetries
                        orientation_masks = set()
                        for symmetry_bit_map in sym_bit_maps:
                            mask = 0
                            for index in canonical_indices:
                                mask |= symmetry_bit_map[index]
                            orientation_masks.add(mask)

                        unique_writer.writerow(
                            [
                                unique_id,
                                canonical_mask,
                                len(orientation_masks),
                                indices_to_coords(canonical_indices),
                            ]
                        )

                    map_writer.writerow([combo_mask, unique_id])

                    if processed % args.progress_every == 0:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0.0
                        ema_rate = rate if ema_rate is None else (ema_alpha * rate + (1 - ema_alpha) * ema_rate)
                        remaining = TOTAL_COMBOS - processed
                        eta = remaining / ema_rate if ema_rate and ema_rate > 0 else 0

                        LOGGER.info(
                            "[%s/%s] %.2f%% | %.0f combos/s | ETA %.1f min",
                            f"{processed:,}",
                            f"{TOTAL_COMBOS:,}",
                            (processed / TOTAL_COMBOS) * 100,
                            rate,
                            eta / 60,
                        )

        total_time = time.time() - start_time
        LOGGER.info("=== DONE ===")
        LOGGER.info("processed: %s", f"{processed:,}")
        LOGGER.info("unique up to symmetry: %s", f"{next_unique_id:,}")
        LOGGER.info("total time: %.1f min", total_time / 60)
        LOGGER.info("avg rate: %.0f combos/s", processed / total_time)
        if args.sort_after:
            LOGGER.info("sorting outputs (may be memory intensive)...")
            sort_csv_by_column(unique_out_path, "canonical_bitmask")
            sort_csv_by_column(map_out_path, "combo_bitmask")

        LOGGER.info("wrote: %s", unique_out_path)
        LOGGER.info("wrote: %s", map_out_path)
    except Exception:
        LOGGER.exception("Failed to generate unique combos")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
