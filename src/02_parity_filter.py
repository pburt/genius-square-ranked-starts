import argparse
import csv
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from board_helpers import classify_by_parity
from path_helpers import DATA_DIR, OUTPUT_DIR, ensure_output_dir

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split unique combos into parity-possible and parity-impossible."
    )
    parser.add_argument(
        "--input",
        default=str(DATA_DIR / "01_unique_combos.csv"),
        help="input CSV path",
    )
    parser.add_argument(
        "--use-output-inputs",
        action="store_true",
        help="read inputs from output/ instead of data/",
    )
    parser.add_argument(
        "--possible-out",
        default=str(OUTPUT_DIR / "02_parity_solvable.csv"),
        help="parity-possible CSV",
    )
    parser.add_argument(
        "--impossible-out",
        default=str(OUTPUT_DIR / "02_parity_unsolvable.csv"),
        help="parity-impossible CSV",
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
        default=2000,
        help="task chunksize for multiprocessing",
    )
    parser.add_argument("--progress-every", type=int, default=100_000, help="progress interval")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        start_time = time.time()
        processed = 0
        possible = 0
        impossible = 0

        ensure_output_dir()
        input_path = (
            OUTPUT_DIR / "01_unique_combos.csv"
            if args.use_output_inputs
            else Path(args.input)
        )
        possible_out_path = Path(args.possible_out)
        impossible_out_path = Path(args.impossible_out)

        worker_count = args.workers if args.workers >= 0 else 0
        if worker_count == 0:
            worker_count = os.cpu_count() or 1
        use_multiprocessing = worker_count > 1

        with (
            open(input_path, "r", newline="") as input_file,
            open(possible_out_path, "w", newline="") as possible_file,
            open(impossible_out_path, "w", newline="") as impossible_file,
        ):
            reader = csv.DictReader(input_file)

            fieldnames = reader.fieldnames or []
            extra_fields = [
                "black_remaining",
                "white_remaining",
                "diff_black_minus_white",
                "parity_possible",
            ]
            out_fields = fieldnames + [field for field in extra_fields if field not in fieldnames]

            possible_writer = csv.DictWriter(possible_file, fieldnames=out_fields)
            impossible_writer = csv.DictWriter(impossible_file, fieldnames=out_fields)
            possible_writer.writeheader()
            impossible_writer.writeheader()

            rows = list(reader)
            masks = [int(row["canonical_bitmask"]) for row in rows]

            if use_multiprocessing:
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    results = executor.map(
                        classify_by_parity,
                        masks,
                        chunksize=max(args.chunksize, 1),
                    )
                    for row, (parity_possible, black_remaining, white_remaining, diff) in zip(rows, results):
                        processed += 1
                        row["black_remaining"] = str(black_remaining)
                        row["white_remaining"] = str(white_remaining)
                        row["diff_black_minus_white"] = str(diff)
                        row["parity_possible"] = "1" if parity_possible else "0"

                        if parity_possible:
                            possible_writer.writerow(row)
                            possible += 1
                        else:
                            impossible_writer.writerow(row)
                            impossible += 1

                        if processed % args.progress_every == 0:
                            elapsed = time.time() - start_time
                            rate = processed / elapsed if elapsed > 0 else 0.0
                            LOGGER.info(
                                "[%s] %.0f rows/s | parity-possible: %s | parity-impossible: %s",
                                f"{processed:,}",
                                rate,
                                f"{possible:,}",
                                f"{impossible:,}",
                            )
            else:
                for row, mask in zip(rows, masks):
                    processed += 1

                    parity_possible, black_remaining, white_remaining, diff = classify_by_parity(mask)

                    row["black_remaining"] = str(black_remaining)
                    row["white_remaining"] = str(white_remaining)
                    row["diff_black_minus_white"] = str(diff)
                    row["parity_possible"] = "1" if parity_possible else "0"

                    if parity_possible:
                        possible_writer.writerow(row)
                        possible += 1
                    else:
                        impossible_writer.writerow(row)
                        impossible += 1

                    if processed % args.progress_every == 0:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed if elapsed > 0 else 0.0
                        LOGGER.info(
                            "[%s] %.0f rows/s | parity-possible: %s | parity-impossible: %s",
                            f"{processed:,}",
                            rate,
                            f"{possible:,}",
                            f"{impossible:,}",
                        )

        elapsed = time.time() - start_time
        LOGGER.info("=== DONE ===")
        LOGGER.info("read rows: %s", f"{processed:,}")
        LOGGER.info("parity-possible: %s", f"{possible:,}")
        LOGGER.info("parity-impossible: %s", f"{impossible:,}")
        LOGGER.info("time: %.1fs (%.0f rows/s)", elapsed, processed / elapsed if elapsed > 0 else 0.0)
        LOGGER.info("wrote: %s", possible_out_path)
        LOGGER.info("wrote: %s", impossible_out_path)
    except Exception:
        LOGGER.exception("Failed to filter by parity")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
