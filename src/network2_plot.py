import argparse
import os
import sys
import sqlite3
import matplotlib
matplotlib.use("Agg")  # no display available in this environment
import matplotlib.pyplot as plt

# Hyperparameter columns on the `runs` table that summary mode / the
# per-epoch-curve legend are allowed to slice, facet, or vary on.
HYPERPARAM_COLUMNS = ["net_sizes", "n_train", "mini_batch_size", "eta",
                      "lmbda", "lmbda1", "dropout_prob", "patience",
                      "sparsity_threshold", "cost", "weight_init"]

SHORT_NAMES = {"net_sizes": "arch", "n_train": "n_train", "mini_batch_size": "mb",
               "eta": "eta", "lmbda": "lmbda", "lmbda1": "lmbda1",
               "dropout_prob": "dropout", "patience": "patience",
               "sparsity_threshold": "sp_thresh", "cost": "cost",
               "weight_init": "winit"}

def parse_run_ids(s):
    return [int(x) for x in s.split(",")]

parser = argparse.ArgumentParser()
parser.add_argument("--db", type=str, default=None,
                     help="Path to the SQLite db; defaults to network2_experiments.db next to this script")
parser.add_argument("--list", action="store_true",
                     help="List all runs in the db and exit")
parser.add_argument("--run_ids", type=parse_run_ids, default=None,
                     help="Comma-separated run_id(s) to plot; defaults to the most recent run")
parser.add_argument("--group_tag", type=str, default=None,
                     help="Plot every run sharing this group_tag (as printed by network2_test.py "
                          "for a sweep). Matches an exact group_tag, OR any group_tag that starts "
                          "with '<value>-' -- this covers a sweep that was split across several "
                          "parallel network2_test.py invocations, where each invocation appends "
                          "its own unique timestamp/random suffix to the group_tag you passed in, "
                          "so the literal strings differ even though they're the same logical "
                          "experiment. Takes precedence over --run_ids.")
parser.add_argument("--plot_last_n_epochs", type=int, default=None,
                     help="Only plot the last N epochs of each run; omit to plot all epochs")
parser.add_argument("--out", type=str, default=None,
                     help="Output PNG path; defaults to network2_plot_runs_<ids>.png in the current "
                          "directory (or network2_plot_summary_<metric>_vs_<x>.png in summary mode)")
parser.add_argument("--summary_x", type=str, default=None, choices=HYPERPARAM_COLUMNS,
                     help="Switches to summary mode: plot one point per run, with this "
                          "hyperparameter on the x-axis, instead of per-epoch curves. Use this "
                          "for a scan (e.g. dropout_prob, lmbda, lmbda1, n_train) where what "
                          "you want to see is 'final/best metric vs swept parameter', not 16 "
                          "raw epoch curves on top of each other.")
parser.add_argument("--summary_metric", type=str, default="eval_accuracy",
                     choices=["eval_accuracy", "train_accuracy", "gap", "weight_norm", "weight_sparsity"],
                     help="Summary mode only: metric for the y-axis. 'gap' is "
                          "train_accuracy - eval_accuracy.")
parser.add_argument("--summary_point", type=str, default="final", choices=["final", "best"],
                     help="Summary mode only: read the metric at the last epoch trained "
                          "('final'), or at the epoch where eval_accuracy peaked ('best' -- "
                          "this is what an early-stopped run would actually return, and is the "
                          "point to use for e.g. dropout, which needs enough epochs and is "
                          "noisy near the end). Whichever metric you pick, it's read from that "
                          "SAME epoch, not independently re-maximized per metric.")
parser.add_argument("--facet_by", type=str, default=None, choices=HYPERPARAM_COLUMNS,
                     help="Summary mode only: split into one subplot per distinct value of this "
                          "hyperparameter (e.g. net_sizes, for a 2D sweep like dropout x capacity).")
args = parser.parse_args()

db_path = args.db
if db_path is None:
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "network2_experiments.db")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

if args.list:
    cursor.execute("""
        SELECT run_id, created_at, group_tag, actual_epochs, eta, lmbda, lmbda1, dropout_prob, patience, run_tag
        FROM runs ORDER BY run_id
    """)
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        print "No runs found in {}".format(db_path)
        sys.exit(0)
    print "run_id\tcreated_at\t\t\tgroup_tag\t\tepochs\teta\tlmbda\tlmbda1\tdropout\tpatience\trun_tag"
    for run_id, created_at, group_tag, actual_epochs, eta, lmbda, lmbda1, dropout_prob, patience, run_tag in rows:
        print "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            run_id, created_at, group_tag, actual_epochs, eta, lmbda, lmbda1, dropout_prob, patience, run_tag)
    sys.exit(0)

if args.group_tag is not None:
    cursor.execute("SELECT run_id FROM runs WHERE group_tag = ? OR group_tag LIKE ? ORDER BY run_id",
                   (args.group_tag, args.group_tag + "-%"))
    run_ids = [row[0] for row in cursor.fetchall()]
    if not run_ids:
        conn.close()
        print "No runs found with group_tag={} (or prefix {}-) in {}".format(
            args.group_tag, args.group_tag, db_path)
        sys.exit(1)
elif args.run_ids is not None:
    run_ids = args.run_ids
else:
    cursor.execute("SELECT MAX(run_id) FROM runs")
    max_run_id = cursor.fetchone()[0]
    if max_run_id is None:
        conn.close()
        print "No runs found in {}".format(db_path)
        sys.exit(1)
    run_ids = [max_run_id]

runs = []
for run_id in run_ids:
    cursor.execute("""
        SELECT run_tag, n_train, n_validation, sparsity_threshold,
               net_sizes, mini_batch_size, eta, lmbda, lmbda1, dropout_prob, patience, cost, weight_init
        FROM runs WHERE run_id = ?
    """, (run_id,))
    row = cursor.fetchone()
    if row is None:
        conn.close()
        print "run_id {} not found in {}".format(run_id, db_path)
        sys.exit(1)
    (run_tag, n_train, n_validation, sparsity_threshold,
     net_sizes, mini_batch_size, eta, lmbda, lmbda1, dropout_prob, patience, cost, weight_init) = row

    cursor.execute("""
        SELECT epoch, training_cost, training_accuracy, evaluation_cost,
               evaluation_accuracy, weight_norm, weight_sparsity
        FROM epochs WHERE run_id = ? ORDER BY epoch
    """, (run_id,))
    epoch_rows = cursor.fetchall()

    epochs = [r[0] for r in epoch_rows]
    training_cost = [r[1] for r in epoch_rows]
    training_accuracy = [r[2] for r in epoch_rows]
    evaluation_cost = [r[3] for r in epoch_rows]
    evaluation_accuracy = [r[4] for r in epoch_rows]
    weight_norm = [r[5] for r in epoch_rows]
    weight_sparsity = [r[6] for r in epoch_rows]

    if args.plot_last_n_epochs is None:
        plot_slice = slice(0, len(epochs))
    else:
        plot_slice = slice(max(0, len(epochs) - args.plot_last_n_epochs), len(epochs))

    runs.append({
        "run_id": run_id,
        "run_tag": run_tag,
        "sparsity_threshold": sparsity_threshold,
        "net_sizes": net_sizes,
        "n_train": n_train,
        "mini_batch_size": mini_batch_size,
        "eta": eta,
        "lmbda": lmbda,
        "lmbda1": lmbda1,
        "dropout_prob": dropout_prob,
        "patience": patience,
        "cost": cost,
        "weight_init": weight_init,
        "epochs": epochs[plot_slice],
        "training_cost": training_cost[plot_slice],
        "training_accuracy": [float(a) / n_train for a in training_accuracy[plot_slice]],
        "evaluation_cost": evaluation_cost[plot_slice],
        "evaluation_accuracy": [float(a) / n_validation for a in evaluation_accuracy[plot_slice]],
        "weight_norm": weight_norm[plot_slice],
        "weight_sparsity": weight_sparsity[plot_slice],
    })

conn.close()


def detect_varying_params(runs):
    varying = []
    for col in HYPERPARAM_COLUMNS:
        if len(set(run[col] for run in runs)) > 1:
            varying.append(col)
    return varying


def build_label(run, varying_params):
    if not varying_params:
        return "run{}".format(run["run_id"])
    parts = ["{}={}".format(SHORT_NAMES.get(col, col), run[col]) for col in varying_params]
    return "{} (run{})".format(",".join(parts), run["run_id"])


if args.summary_x is not None:
    # --- Summary mode: one point per run (at a chosen epoch), x = swept
    # hyperparameter, optionally faceted into subplots by a second
    # hyperparameter. This is the view you want for a scan -- "metric vs
    # swept parameter" -- rather than N raw epoch curves stacked on top of
    # each other.
    if args.facet_by == args.summary_x:
        print "--facet_by and --summary_x can't be the same column"
        sys.exit(1)

    def metric_at(run, index):
        if args.summary_metric == "eval_accuracy":
            return run["evaluation_accuracy"][index]
        elif args.summary_metric == "train_accuracy":
            return run["training_accuracy"][index]
        elif args.summary_metric == "gap":
            return run["training_accuracy"][index] - run["evaluation_accuracy"][index]
        elif args.summary_metric == "weight_norm":
            return run["weight_norm"][index]
        elif args.summary_metric == "weight_sparsity":
            return run["weight_sparsity"][index]

    points = []
    for run in runs:
        if not run["evaluation_accuracy"]:
            continue
        if args.summary_point == "best":
            index = run["evaluation_accuracy"].index(max(run["evaluation_accuracy"]))
        else:
            index = len(run["evaluation_accuracy"]) - 1
        x_val = run[args.summary_x]
        facet_val = run[args.facet_by] if args.facet_by else None
        points.append({
            "x": x_val,
            "x_key": round(x_val, 6) if isinstance(x_val, float) else x_val,
            "y": metric_at(run, index),
            "facet": facet_val,
            "run_id": run["run_id"],
        })

    if not points:
        print "No runs with data to summarize"
        sys.exit(1)

    facets = sorted(set(p["facet"] for p in points), key=lambda v: (v is None, v)) if args.facet_by else [None]
    n_facets = len(facets)
    ncols = 2 if n_facets > 1 else 1
    nrows = (n_facets + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for ax, facet in zip(axes_flat, facets):
        facet_points = [p for p in points if p["facet"] == facet]
        by_x = {}
        for p in facet_points:
            by_x.setdefault(p["x_key"], []).append(p)
        xs_sorted = sorted(by_x.keys())
        means = [sum(p["y"] for p in by_x[x]) / len(by_x[x]) for x in xs_sorted]

        for x in xs_sorted:
            group = by_x[x]
            if len(group) > 1:
                ax.scatter([x] * len(group), [p["y"] for p in group],
                           color="tab:blue", alpha=0.4, zorder=2)
        ax.plot(xs_sorted, means, color="tab:blue", marker="o", zorder=3)

        ax.set_xlabel(SHORT_NAMES.get(args.summary_x, args.summary_x))
        ax.set_ylabel(args.summary_metric)
        if args.facet_by:
            ax.set_title("{}={}".format(SHORT_NAMES.get(args.facet_by, args.facet_by), facet))
        else:
            ax.set_title("{} vs {}".format(args.summary_metric, args.summary_x))

    for ax in axes_flat[len(facets):]:
        ax.axis("off")

    plt.tight_layout()
    out_path = args.out
    if out_path is None:
        out_path = "network2_plot_summary_{}_vs_{}.png".format(args.summary_metric, args.summary_x)
    plt.savefig(out_path)
    print "Saved summary plot ({} points, {} point) to {}".format(
        len(points), args.summary_point, out_path)
    sys.exit(0)


# --- Default mode: overlay full per-epoch curves for each run. ---
varying_params = detect_varying_params(runs)

fig, ((ax_cost, ax_acc), (ax_norm, ax_sparsity)) = plt.subplots(2, 2, figsize=(10, 8))

for run in runs:
    label = build_label(run, varying_params)
    line, = ax_cost.plot(run["epochs"], run["training_cost"], label="{} train".format(label))
    ax_cost.plot(run["epochs"], run["evaluation_cost"], linestyle="--", color=line.get_color(),
                 label="{} eval".format(label))
ax_cost.set_xlabel("Epoch")
ax_cost.set_ylabel("Cost")
ax_cost.set_title("Cost per epoch")
ax_cost.legend()

for run in runs:
    label = build_label(run, varying_params)
    line, = ax_acc.plot(run["epochs"], run["training_accuracy"], label="{} train".format(label))
    ax_acc.plot(run["epochs"], run["evaluation_accuracy"], linestyle="--", color=line.get_color(),
                label="{} eval".format(label))
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_title("Accuracy per epoch")
ax_acc.legend()

for run in runs:
    ax_norm.plot(run["epochs"], run["weight_norm"], label=build_label(run, varying_params))
ax_norm.set_xlabel("Epoch")
ax_norm.set_ylabel("Norm")
ax_norm.set_title("Weight norm per epoch")
ax_norm.legend()

for run in runs:
    ax_sparsity.plot(run["epochs"], run["weight_sparsity"], label=build_label(run, varying_params))
ax_sparsity.set_xlabel("Epoch")
sparsity_thresholds = set(run["sparsity_threshold"] for run in runs)
if len(sparsity_thresholds) == 1:
    ax_sparsity.set_ylabel("Fraction |w| < {}".format(list(sparsity_thresholds)[0]))
else:
    ax_sparsity.set_ylabel("Fraction |w| below threshold")
ax_sparsity.set_title("Weight sparsity per epoch")
ax_sparsity.legend()

plt.tight_layout()

out_path = args.out
if out_path is None:
    ids_str = "-".join(str(run["run_id"]) for run in runs)
    out_path = "network2_plot_runs_{}.png".format(ids_str)

plt.savefig(out_path)
print "Saved plot to {}".format(out_path)
