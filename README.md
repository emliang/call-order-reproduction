# Paired call-order experiment

This package reproduces the affine-ReLU comparison in Appendix E. The
two orders have identical parameter counts and call each module the
same number of times. Order A repeats the first module before visiting
the others. Order B visits every module before repeating the first.
Each pair of fits shares its initial weights and training batches.

The experiment uses 100-peak Lipschitz targets in two and three
dimensions at module/call budgets (4, 16), (5, 25), and (6, 36). Each
setting has three targets and five training seeds per target.

## Recreate the table

Only Python is needed to aggregate the saved results.

~~~sh
python reproduce.py --from-records --output table-from-records
~~~

The output is `mean_errors.csv`. It averages per-fit RMSE and maximum
absolute error over the 15 fits for each order and setting. The test
maximum is over the evaluation grid. Multiply the error columns by
1,000 to obtain the units used in Appendix E.

## Train the models

~~~sh
python -m pip install -r requirements.txt
python reproduce.py --quick --output quick-run
python reproduce.py --jobs 9 --output full-run
~~~

The quick run trains one pair per dimension at (4, 16). Both commands
use 10,000 Adam updates per fit. The full run trains all 180 models.
Workers are single-threaded by default. Each run writes per-fit
checkpoint metrics to `runs.json` and aggregate errors to `mean_errors.csv`.
Use a new directory for each run.

The archived results were generated with CPython 3.10.20 and PyTorch
2.11.0 on an Apple Silicon CPU. To require exact agreement with the saved
checkpoint metrics, add `--verify`. On other CPU environments, training
can produce small numerical differences.

~~~sh
python reproduce.py --quick --verify --output verified-quick-run
~~~

## Files

- `experiment.py` defines the targets, model, training loop, and metrics.
- `reproduce.py` trains the model pairs or aggregates saved metrics.
- `data/config.json` contains the experimental protocol and target definitions.
- `data/recorded_results.json` contains the saved checkpoint metrics.
