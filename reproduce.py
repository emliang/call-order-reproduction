"""Run the paired training comparison or tabulate its saved results."""

import argparse
import csv
import json
import platform
from concurrent.futures import ProcessPoolExecutor
from math import sqrt
from pathlib import Path
from statistics import mean

DATA = Path(__file__).resolve().parent / "data"
METRICS = ("train_rmse", "train_max_abs_error", "test_grid_rmse", "test_grid_max_abs_error")


def read_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def train(config, jobs, quick):
    from experiment import configure_worker, train_target

    settings = config["training"]
    seeds = settings["seeds"][:1] if quick else settings["seeds"]
    budgets = config["budgets"][:1] if quick else config["budgets"]
    with ProcessPoolExecutor(max_workers=jobs, initializer=configure_worker,
                             initargs=(settings["threads"],)) as pool:
        scheduled = []
        for dimension, problem in config["problems"].items():
            d = int(dimension)
            targets = problem["targets"][:1] if quick else problem["targets"]
            for modules, calls in budgets:
                tasks = [
                    pool.submit(train_target, d, modules, calls, problem["grid_side"],
                                target, seeds, settings)
                    for target in targets
                ]
                scheduled.append((d, modules, calls, tasks))

        results = []
        for d, modules, calls, tasks in scheduled:
            targets = []
            for task in tasks:
                target = task.result()
                targets.append(target)
                print(f"Finished d={d}, J={modules}, T={calls}, target={target['id']}", flush=True)
            results.append({"dimension": d, "modules": modules, "calls": calls, "targets": targets})
    return results


def setting_key(result):
    return result["dimension"], result["modules"], result["calls"]


def verify_checkpoints(results, saved_results):
    saved_by_setting = {setting_key(result): result["targets"] for result in saved_results}
    for result in results:
        d, modules, calls = setting_key(result)
        saved_targets = {target["id"]: target["pairs"]
                         for target in saved_by_setting[d, modules, calls]}
        for target in result["targets"]:
            saved_pairs = {pair["seed"]: pair for pair in saved_targets[target["id"]]}
            for pair in target["pairs"]:
                seed = pair["seed"]
                if pair != saved_pairs[seed]:
                    raise RuntimeError(
                        f"Checkpoint mismatch for d={d}, J={modules}, T={calls}, "
                        f"target={target['id']}, seed={seed}."
                    )
    print("All checkpoint metrics match the saved records exactly.", flush=True)


def write_means(results, step, path):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dimension", "modules", "calls", "order", "fits", *METRICS))
        for result in results:
            pairs = [pair for target in result["targets"] for pair in target["pairs"]]
            for order in ("A", "B"):
                errors = [pair[order][step] for pair in pairs]
                train_rmse = mean(sqrt(error["train_mse"]) for error in errors)
                train_max = mean(error["train_max_abs_error"] for error in errors)
                test_rmse = mean(sqrt(error["grid"]["mse"]) for error in errors)
                test_max = mean(error["grid"]["max_abs_error"] for error in errors)
                writer.writerow((*setting_key(result), order, len(errors),
                                 train_rmse, train_max, test_rmse, test_max))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--from-records", action="store_true", help="Tabulate the saved results.")
    mode.add_argument("--quick", action="store_true", help="Train one pair in each dimension.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of CPU workers.")
    parser.add_argument("--output", type=Path, required=True, help="New output directory.")
    parser.add_argument("--verify", action="store_true", help="Require an exact checkpoint match.")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive.")
    if args.from_records and args.verify:
        parser.error("--verify applies to fresh training runs.")
    if args.output.exists():
        parser.error("--output must be a new directory.")

    config = read_json("config.json")
    if args.from_records:
        results = read_json("recorded_results.json")
    else:
        results = train(config, args.jobs, args.quick)
        if args.verify:
            verify_checkpoints(results, read_json("recorded_results.json"))

    args.output.mkdir(parents=True)
    if not args.from_records:
        import torch

        environment = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": f"{platform.machine()} CPU",
        }
        (args.output / "runs.json").write_text(
            json.dumps({"environment": environment, "results": results}, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    step = str(config["training"]["checkpoints"][-1])
    write_means(results, step, args.output / "mean_errors.csv")
    print(f"Wrote {args.output / 'mean_errors.csv'}")


if __name__ == "__main__":
    main()
