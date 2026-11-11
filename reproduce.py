"""Run the paired training comparison or tabulate its saved results."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from math import sqrt
from pathlib import Path
import platform
from statistics import mean


DATA = Path(__file__).resolve().parent / "data"
METRICS = ("train_rmse", "train_max_abs_error", "test_grid_rmse", "test_grid_max_abs_error")


def read_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def train(config, records, jobs, quick, allow_differences):
    import torch
    from experiment import configure_worker, train_target

    runtime = {
        "python": platform.python_version(), "torch": torch.__version__,
        "device": f"{platform.machine()} CPU",
    }
    if runtime != config["reference_environment"] and not allow_differences:
        raise SystemExit("Use the reference CPU environment in README.md for exact replay, "
                         "or pass --allow-differences for another CPU runtime.")
    settings = config["training"]
    seeds = settings["seeds"][:1] if quick else settings["seeds"]
    expected = {(r["dimension"], r["modules"], r["calls"]): r["targets"] for r in records}
    budgets = config["budgets"][:1] if quick else config["budgets"]
    results = []
    with ProcessPoolExecutor(max_workers=jobs, initializer=configure_worker,
                             initargs=(settings["threads"],)) as pool:
        pending = {}
        for dimension, problem in config["problems"].items():
            d = int(dimension)
            targets = problem["targets"][:1] if quick else problem["targets"]
            for modules, calls in budgets:
                key = (d, modules, calls)
                observed = [None] * len(targets)
                results.append({"dimension": d, "modules": modules, "calls": calls, "targets": observed})
                for index, target in enumerate(targets):
                    reference = expected[key][index]
                    assert target["id"] == reference["id"]
                    reference = {"id": reference["id"], "pairs": reference["pairs"][:len(seeds)]}
                    future = pool.submit(train_target, d, modules, calls, problem, target, seeds, settings)
                    pending[future] = (key, index, observed, reference)

        for future in as_completed(pending):
            (d, modules, calls), index, observed, reference = pending[future]
            result = future.result()
            name = f"d={d}, J={modules}, T={calls}, target={result['id']}"
            exact = result == reference
            if not exact and not allow_differences:
                raise RuntimeError(f"Checkpoint metrics differ for {name}.")
            observed[index] = result
            print(f"{name}: {'exact match' if exact else 'numerical differences'}", flush=True)
    return runtime, results


def write_means(results, step, path):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dimension", "modules", "calls", "order", "fits", *METRICS))
        for result in results:
            pairs = [pair for target in result["targets"] for pair in target["pairs"]]
            for order in ("A", "B"):
                errors = [pair[order][step] for pair in pairs]
                metrics = [
                    (sqrt(f["train_mse"]), f["train_max_abs_error"],
                     sqrt(f["grid"]["mse"]), f["grid"]["max_abs_error"])
                    for f in errors
                ]
                writer.writerow((result["dimension"], result["modules"], result["calls"],
                                 order, len(errors), *(mean(values) for values in zip(*metrics))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--from-records", action="store_true", help="Compute the table without training.")
    mode.add_argument("--quick", action="store_true", help="Train one saved pair in each dimension.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel single-threaded CPU workers.")
    parser.add_argument("--output", type=Path, required=True, help="New output directory.")
    parser.add_argument("--allow-differences", action="store_true",
                        help="Train on another CPU runtime and report differences from the saved runs.")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive.")
    if args.from_records and args.allow_differences:
        parser.error("--allow-differences applies to training.")
    if args.output.exists():
        parser.error("--output must be a new directory.")
    config = read_json("config.json")
    records = read_json("recorded_results.json")
    args.output.mkdir(parents=True)
    if not args.from_records:
        runtime, records = train(config, records, args.jobs, args.quick, args.allow_differences)
        (args.output / "runs.json").write_text(
            json.dumps({"environment": runtime, "results": records}, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    step = str(config["training"]["checkpoints"][-1])
    write_means(records, step, args.output / "mean_errors.csv")
    print(f"Wrote {args.output / 'mean_errors.csv'}", flush=True)


if __name__ == "__main__":
    main()
