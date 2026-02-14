import argparse
import csv
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional

from board_helpers import count_solutions
from path_helpers import DATA_DIR, OUTPUT_DIR, ensure_output_dir

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve boards from a CSV and split into solvable and unsolvable outputs."
    )
    parser.add_argument(
        "--input",
        default=str(DATA_DIR / "02_parity_solvable.csv"),
        help="input CSV path",
    )
    parser.add_argument(
        "--use-output-inputs",
        action="store_true",
        help="read inputs from output/ instead of data/",
    )
    parser.add_argument(
        "--solvable-out",
        default=str(OUTPUT_DIR / "03_solvable.csv"),
        help="solvable CSV output",
    )
    parser.add_argument(
        "--unsolvable-out",
        default=str(OUTPUT_DIR / "03_unsolvable.csv"),
        help="unsolvable CSV output",
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
        default=100,
        help="task chunksize for multiprocessing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="rows to buffer per batch (limits memory usage)",
    )
    parser.add_argument("--progress-every", type=int, default=1_000, help="progress interval")
    parser.add_argument("--max-solutions", type=int, default=1, help="early exit after N solutions (0 means full count)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        max_solutions: Optional[int] = args.max_solutions if args.max_solutions and args.max_solutions > 0 else None

        start_time = time.time()
        processed = 0
        solvable = 0
        unsolvable = 0

        ensure_output_dir()
        input_path = (
            OUTPUT_DIR / "02_parity_solvable.csv"
            if args.use_output_inputs
            else Path(args.input)
        )
        solvable_out_path = Path(args.solvable_out)
        unsolvable_out_path = Path(args.unsolvable_out)

        worker_count = args.workers if args.workers >= 0 else 0
        if worker_count == 0:
            worker_count = os.cpu_count() or 1
        use_multiprocessing = worker_count > 1

        with open(input_path, "r", newline="") as input_file:
            total = sum(1 for _ in input_file) - 1

        with (
            open(input_path, "r", newline="") as input_file,
            open(solvable_out_path, "w", newline="") as solvable_file,
            open(unsolvable_out_path, "w", newline="") as unsolvable_file,
        ):
            reader = csv.DictReader(input_file)
            fields = list(reader.fieldnames or [])
            extra_fields = ["solution_count"]
            for field in extra_fields:
                if field not in fields:
                    fields.append(field)

            solvable_writer = csv.DictWriter(solvable_file, fieldnames=fields)
            unsolvable_writer = csv.DictWriter(unsolvable_file, fieldnames=fields)
            solvable_writer.writeheader()
            unsolvable_writer.writeheader()

            def iter_batches():
                batch = []
                for row in reader:
                    batch.append(row)
                    if len(batch) >= args.batch_size:
                        yield batch
                        batch = []
                if batch:
                    yield batch

            if use_multiprocessing:
                with ProcessPoolExecutor(max_workers=worker_count) as executor:
                    solver = partial(count_solutions, max_solutions=max_solutions)
                    for batch in iter_batches():
                        masks = [int(row["canonical_bitmask"]) for row in batch]
                        results = executor.map(
                            solver,
                            masks,
                            chunksize=max(args.chunksize, 1),
                        )
                        for row, solution_count in zip(batch, results):
                            processed += 1
                            row["solution_count"] = str(solution_count)

                            if solution_count > 0:
                                solvable_writer.writerow(row)
                                solvable += 1
                            else:
                                unsolvable_writer.writerow(row)
                                unsolvable += 1

                            if processed % args.progress_every == 0:
                                elapsed = time.time() - start_time
                                rate = processed / elapsed if elapsed > 0 else 0.0
                                eta = (total - processed) / rate if rate else 0
                                LOGGER.info(
                                    "[%s/%s] %.2f%% | %.2f boards/s | ETA %.1f min | solvable %s / unsolvable %s",
                                    f"{processed:,}",
                                    f"{total:,}",
                                    (processed / total) * 100 if total > 0 else 0.0,
                                    rate,
                                    eta / 60,
                                    f"{solvable:,}",
                                    f"{unsolvable:,}",
                                )
            else:
                for batch in iter_batches():
                    for row in batch:
                        processed += 1
                        canonical_mask = int(row["canonical_bitmask"])
                        solution_count = count_solutions(canonical_mask, max_solutions=max_solutions)

                        row["solution_count"] = str(solution_count)

                        if solution_count > 0:
                            solvable_writer.writerow(row)
                            solvable += 1
                        else:
                            unsolvable_writer.writerow(row)
                            unsolvable += 1

                        if processed % args.progress_every == 0:
                            elapsed = time.time() - start_time
                            rate = processed / elapsed if elapsed > 0 else 0.0
                            eta = (total - processed) / rate if rate else 0
                            LOGGER.info(
                                "[%s/%s] %.2f%% | %.2f boards/s | ETA %.1f min | solvable %s / unsolvable %s",
                                f"{processed:,}",
                                f"{total:,}",
                                (processed / total) * 100 if total > 0 else 0.0,
                                rate,
                                eta / 60,
                                f"{solvable:,}",
                                f"{unsolvable:,}",
                            )

        elapsed = time.time() - start_time
        LOGGER.info("=== DONE ===")
        LOGGER.info("processed: %s", f"{processed:,}")
        LOGGER.info("solvable: %s", f"{solvable:,}")
        LOGGER.info("unsolvable: %s", f"{unsolvable:,}")
        LOGGER.info("time: %.1f min", elapsed / 60)
    except Exception:
        LOGGER.exception("Failed to solve boards")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
