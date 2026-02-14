import argparse
import csv
import logging
import os
import statistics
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional

from board_helpers import count_solutions
from path_helpers import DATA_DIR, OUTPUT_DIR, ensure_output_dir

LOGGER = logging.getLogger(__name__)


def _count_for_mask(mask: int) -> int:
    return count_solutions(mask, max_solutions=None)


def _open_output(path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _sort_output_by_solution_count(output_path: Path) -> None:
    """Sort the output CSV by solution_count ascending (in-memory)."""
    with output_path.open("r", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    rows.sort(key=lambda row: int(row.get("solution_count", "0")))

    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recount total solutions for solvable boards and update solution_count column."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DATA_DIR / "03_solvable.csv"),
        help="input CSV path",
    )
    parser.add_argument(
        "--use-output-inputs",
        action="store_true",
        help="read inputs from output/ instead of data/",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "04_solvable_with_solution_counts.csv"),
        help="output CSV path",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="progress interval",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="number of worker processes (0 = cpu count)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100,
        help="task chunksize for multiprocessing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse solution_count from existing output file when possible",
    )
    parser.add_argument(
        "--fsync-every",
        type=int,
        default=500,
        help="force fsync every N written rows (0 to disable)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        ensure_output_dir()
        input_path = (
            OUTPUT_DIR / "03_solvable.csv"
            if args.use_output_inputs
            else Path(args.input)
        )
        output_path = Path(args.output)

        with open(input_path, "r", newline="") as input_file:
            total = max(sum(1 for _ in input_file) - 1, 0)

        start_time = time.time()
        processed = 0

        counts: List[int] = []
        min_count: Optional[int] = None
        max_count: Optional[int] = None
        total_count = 0

        _open_output(output_path)

        with open(input_path, "r", newline="") as input_file:
            reader = csv.DictReader(input_file)
            fields = list(reader.fieldnames or [])
            extra_fields = ["solution_count"]
            for field in extra_fields:
                if field not in fields:
                    fields.append(field)
            rows: List[dict] = list(reader)

        existing_counts: Dict[int, int] = {}
        if args.resume and os.path.exists(output_path):
            with open(output_path, "r", newline="") as output_file:
                existing_reader = csv.DictReader(output_file)
                for row in existing_reader:
                    mask = _parse_int(row.get("canonical_bitmask"))
                    count = _parse_int(row.get("solution_count"))
                    if mask is None or count is None:
                        continue
                    existing_counts[mask] = count

        rows_existing: List[dict] = []
        rows_missing: List[dict] = []
        for row in rows:
            mask = int(row["canonical_bitmask"])
            existing = existing_counts.get(mask)
            if existing is not None:
                row["solution_count"] = str(existing)
                rows_existing.append(row)
            else:
                rows_missing.append(row)

        file_has_data = os.path.exists(output_path) and os.path.getsize(output_path) > 0
        write_mode = "a" if args.resume and file_has_data else "w"
        with open(output_path, write_mode, newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fields)
            if write_mode == "w":
                writer.writeheader()

            worker_count = args.workers if args.workers and args.workers > 0 else 0
            if worker_count == 0:
                worker_count = os.cpu_count() or 1
            use_multiprocessing = worker_count > 1
            executor: Optional[ProcessPoolExecutor] = None

            try:
                written = 0

                # Write existing rows first when starting a new file
                if write_mode == "w":
                    for row in rows_existing:
                        processed += 1
                        writer.writerow(row)
                        written += 1
                        count = int(row["solution_count"])
                        counts.append(count)
                        total_count += count
                        min_count = count if min_count is None else min(min_count, count)
                        max_count = count if max_count is None else max(max_count, count)

                        if args.progress_every and processed % args.progress_every == 0:
                            elapsed = time.time() - start_time
                            rate = processed / elapsed if elapsed > 0 else 0.0
                            eta = (total - processed) / rate if rate else 0
                            LOGGER.info(
                                "[%s/%s] %.2f%% | %.2f boards/s | ETA %.1f min",
                                f"{processed:,}",
                                f"{total:,}",
                                (processed / total) * 100 if total > 0 else 0.0,
                                rate,
                                eta / 60,
                            )

                # Compute missing rows with completion-order streaming
                if rows_missing:
                    if use_multiprocessing:
                        executor = ProcessPoolExecutor(max_workers=worker_count)
                        masks = [int(row["canonical_bitmask"]) for row in rows_missing]
                        results = executor.map(
                            _count_for_mask,
                            masks,
                            chunksize=max(args.chunksize, 1),
                        )
                    else:
                        results = (
                            _count_for_mask(int(row["canonical_bitmask"]))
                            for row in rows_missing
                        )

                    for row, count in zip(rows_missing, results):
                        row["solution_count"] = str(count)

                        writer.writerow(row)
                        written += 1
                        output_file.flush()
                        if args.fsync_every and written % args.fsync_every == 0:
                            os.fsync(output_file.fileno())

                        processed += 1
                        counts.append(count)
                        total_count += count
                        min_count = count if min_count is None else min(min_count, count)
                        max_count = count if max_count is None else max(max_count, count)

                        if args.progress_every and processed % args.progress_every == 0:
                            elapsed = time.time() - start_time
                            rate = processed / elapsed if elapsed > 0 else 0.0
                            eta = (total - processed) / rate if rate else 0
                            LOGGER.info(
                                "[%s/%s] %.2f%% | %.2f boards/s | ETA %.1f min",
                                f"{processed:,}",
                                f"{total:,}",
                                (processed / total) * 100 if total > 0 else 0.0,
                                rate,
                                eta / 60,
                            )
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)

        elapsed = time.time() - start_time
        n = len(counts)
        if n == 0:
            LOGGER.warning("No rows processed.")
            return

        mean = total_count / n
        median = statistics.median(counts)
        stdev = statistics.pstdev(counts) if n > 1 else 0.0

        LOGGER.info("=== DONE ===")
        LOGGER.info("processed: %s", f"{n:,}")
        LOGGER.info("total solutions: %s", f"{total_count:,}")
        LOGGER.info("min solution_count: %s", min_count)
        LOGGER.info("max solution_count: %s", max_count)
        LOGGER.info("mean solution_count: %.3f", mean)
        LOGGER.info("median solution_count: %s", median)
        LOGGER.info("stdev solution_count: %.3f", stdev)
        LOGGER.info("sorting output by solution_count...")
        _sort_output_by_solution_count(output_path)
        LOGGER.info("time: %.1f min", elapsed / 60)
    except Exception:
        LOGGER.exception("Failed to recount solutions")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
