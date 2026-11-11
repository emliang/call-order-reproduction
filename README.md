# Paired call-order experiment

Reproduces the affine-ReLU comparison in Appendix E. Orders A and B use
the same number of parameters and the same call count for each module.
A spends the extra calls to the first module before visiting the other
modules. B visits all modules first.

The experiment uses 100-peak Lipschitz targets in two and three dimensions
and module/call budgets (4, 16), (5, 25), and (6, 36). Each setting has
three targets and five pairs of fits per target, with 10,000 Adam updates
per fit. The target heights, scaling, training settings, and seeds are
in data/config.json. Training inputs and evaluation grids are generated
locally.

## Recreate the table from the saved results

This command uses only the Python standard library.

~~~sh
python reproduce.py --from-records --output table-from-records
~~~

The CSV gives the mean over the 15 fits for each order and setting.
RMSE and maximum absolute error are computed separately for each fit.
The test maximum is taken over the evaluation grid. Multiply the error
columns by 1,000 to match the units in Appendix E.

## Train from scratch

The exact replay environment is CPython 3.10.20 and PyTorch 2.11.0 on
an Apple Silicon CPU. Install PyTorch in a dedicated environment.

~~~sh
python -m pip install -r requirements.txt
python reproduce.py --quick --output quick-run
python reproduce.py --jobs 9 --output full-run
~~~

The quick command runs one pair per dimension at (4, 16) with the full
10,000-step schedule. The full command trains all 180 models. Each
worker uses one CPU thread and deterministic PyTorch operations. The
default is one worker.

The runner checks the fresh checkpoint metrics against the saved
results and writes runs.json and mean_errors.csv to the output
directory. On a different CPU runtime, pass --allow-differences to
use the same protocol and report whether the metrics match. For Linux,
PyTorch offers CPU wheels at https://download.pytorch.org/whl/cpu.

## Files

- experiment.py defines the model, targets, training loop, and metrics.
- reproduce.py runs the experiment or aggregates the saved results.
- data/config.json contains the settings and targets.
- data/recorded_results.json contains the per-fit checkpoint metrics.
