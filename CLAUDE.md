# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is a personal study fork of the code samples for Michael Nielsen's
["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com) book. The `src/` core files
(`network.py`, `network2.py`, `network3.py`, `mnist_loader.py`, etc.) are the book's original Python 2 code,
kept close to upstream style (simple, readable, not optimized — see each file's module docstring). On top of
that, this fork adds an experiment/CLI layer around `network2.py` for working through the book's chapter-3
material (regularization, overfitting, weight initialization) hands-on.

The working goal is to learn the book, not to run a research project. If asked to extend the experiment
tooling further, prefer the book's own style — run each config 3 times (cheap now thanks to `SGD2`'s
`patience` early stopping — see `overfitting-experiments-log.md` section 4.18 for why 3 reps, and which
`patience` value to use: 10 with no regularization, 15 once L2/L1/dropout is added), look at the result,
build intuition — over statistically rigorous multi-repeat/grid-sweep work unless explicitly asked for
that depth.

## Language version and runtime

All code is **Python 2** (`print` statements, `xrange`, `cPickle`). It does **not** run under Python 3.

Training/experiment scripts run inside a Docker container named `py27` (`python:2.7-slim` image, with numpy
and matplotlib installed), already running in the background. The container mounts the project root at `/w`
(bidirectional sync with the WSL-side repo root — no `docker cp` needed). Typical invocation from the WSL
side:

```bash
docker exec py27 sh -c "cd /w/src && python network2_test.py --n_train 1000 --n_epochs 400"
```

For parallel runs (e.g. a parameter sweep split into multiple `docker exec` calls), pin BLAS/OpenMP threads
per process so they don't fight each other for cores:

```bash
docker exec -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 py27 sh -c "..."
```

For a one-off analysis script you don't want to add to the repo, pipe it over stdin rather than `docker cp`
(the container only sees paths under `/w`, not arbitrary host paths):

```bash
cat some_script.py | docker exec -i py27 python -
```

`network3.py` additionally requires Theano 0.6/0.7 (see `requirements.txt`); it has not been updated for
later Theano versions and the upstream README explicitly says it won't be.

## Architecture

- `src/network.py` — Chapter 1/2 network: sigmoid neurons, quadratic cost, plain SGD + backprop. Two
  backprop implementations exist (`backprop`/`update_mini_batch`, per-example; `backprop2`/`update_mini_batch2`,
  vectorized over the whole mini-batch) — `SGD` calls the vectorized path.
- `src/network2.py` — Chapter 3 network: adds `CrossEntropyCost`/`QuadraticCost` cost objects, L2
  regularization, better (`1/sqrt(n_in)`-scaled) weight initialization vs. the naive `large_weight_initializer`
  kept for comparison. `Network.SGD` (`SGD2` internally) is the main entry point and supports early stopping
  via a `patience` parameter: it snapshots weights/biases whenever validation accuracy improves and rolls back
  to the best snapshot at the end of training — the returned network is always the best-seen one, not the
  last epoch. Note the per-epoch arrays it returns (`evaluation_accuracy`, `weight_norm`, ...) are the *full*
  training record, not truncated at the best epoch — to get the value matching the returned (rolled-back)
  network, index with `np.argmax(evaluation_accuracy)`, not `[-1]`. `patience=10` is enough with no
  regularization; once L2/L1/dropout is in play, use `patience=15` (see `overfitting-experiments-log.md`
  section 4.18 for the backtest behind both numbers).
- `src/network3.py` / `src/conv.py` — Chapter 6 convolutional network built on Theano (conv/pool layers,
  ReLU support). Not yet exercised in this fork's experiment log.
- `src/mnist_loader.py` — loads `data/mnist.pkl.gz` into train/validation/test splits; `load_data_wrapper()`
  is the form the `Network` classes expect. Takes an optional `filename` to load `data/mnist_expanded.pkl.gz`
  instead (see `--training_data` below).
- `src/expand_mnist.py` — one-off data augmentation: displaces each of the 50,000 training images by 1px in
  each of 4 directions to build a 250,000-image set, saved to `data/mnist_expanded.pkl.gz` (gitignored, like
  `mnist.pkl.gz`; regenerate with `python expand_mnist.py` in the `py27` container if missing — a few minutes).
- `src/network2_test.py` — CLI harness around `network2.Network`. Every hyperparameter flag
  (`--net_sizes`, `--n_train`, `--n_epochs`, `--mini_batch_size`, `--eta`, `--lmbda`, `--lmbda1`,
  `--dropout_prob`, `--patience`, `--sparsity_threshold`) takes a comma-separated list (semicolon-separated
  for `--net_sizes` to separate whole architectures, e.g. `784,30,10;784,100,10`); the script runs the full
  Cartesian product of all lists in one invocation and writes one row per run (+ one row per epoch) into
  `src/network2_experiments.db` (SQLite, gitignored). `--cost` (default `cross_entropy`), `--weight_init`
  (default `default`), and `--training_data {mnist,mnist_expanded}` (default `mnist`) are single-value
  choices, *not* part of the sweep — compare them across separate invocations. Because it's a Cartesian product,
  `--lmbda` and `--lmbda1` cannot be swept as paired tuples in one call — sweep one dimension at a time if
  you need specific (λ, λ1) pairs. Every invocation gets a `group_tag` (your `--group_tag` prefix, if given,
  plus an always-appended timestamp+random suffix) for later batch retrieval; query by prefix (`LIKE
  'tag-%'`), not exact match, since parallel invocations get different suffixes.
- `src/network2_plot.py` — reads `network2_experiments.db` and plots; fully decoupled from the training
  script (no PNGs are generated during training). Default mode overlays per-epoch curves
  (cost/accuracy/weight_norm/sparsity) for a set of runs. `--summary_x <param>` switches to summary mode:
  final-or-best metric vs. the swept parameter (`--summary_metric {eval_accuracy,train_accuracy,gap,
  weight_norm,weight_sparsity}`, `--summary_point {final,best}`), optionally faceted with `--facet_by
  <param>` for 2D sweeps.
- `src/network2_early_stop_backtest.py` — standalone backtest of the early-stopping logic.
- `src/toy_cost_demo.py` — minimal single-neuron demo reproducing the book's quadratic-vs-cross-entropy
  "starts off badly wrong, learns slowly" illustration.
- `fig/` — standalone plotting/demo scripts (one file per figure, mostly matplotlib) used to generate the
  book's figures; independent of the `src/` network code.

## Experiment notes

- [overfitting-notes.md](overfitting-notes.md) — theory notes (in Chinese) on diagnosing overfitting:
  train/val loss divergence as the primary signal, why loss leads accuracy, weight-norm growth as a
  structural indicator.
- [overfitting-experiments-log.md](overfitting-experiments-log.md) — session-handoff log (in Chinese) of
  actual experiments run, the current state of the code/data, and the outstanding chapter-3 TODOs (softmax +
  log-likelihood output layer, momentum SGD, tanh/ReLU neurons — data augmentation is done, see section
  4.16). Read this first when resuming experiment work — it records the current direction (follow the book
  chapter-by-chapter, 3-repeat intuition over rigorous grid sweeps) and avoids re-deriving context from
  scratch.

## Data

`data/mnist.pkl.gz` is the full MNIST dataset pickle the book's loader expects — already present, don't
regenerate it. `data/mnist_expanded.pkl.gz` is the augmented 250,000-image training set built by
`src/expand_mnist.py` (see Architecture above); regenerate it the same way if missing. `src/network2_experiments.db`
and `*.pkl`/`*.db` files are gitignored; don't expect them to be committed or to survive a fresh clone.
