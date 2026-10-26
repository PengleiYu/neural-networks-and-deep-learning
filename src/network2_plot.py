import argparse
import os
import sys
import sqlite3
import matplotlib
matplotlib.use("Agg")  # no display available in this environment
import matplotlib.pyplot as plt

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
                          "for a sweep); takes precedence over --run_ids")
parser.add_argument("--plot_last_n_epochs", type=int, default=None,
                     help="Only plot the last N epochs of each run; omit to plot all epochs")
parser.add_argument("--out", type=str, default=None,
                     help="Output PNG path; defaults to network2_plot_runs_<ids>.png in the current directory")
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
    cursor.execute("SELECT run_id FROM runs WHERE group_tag = ? ORDER BY run_id", (args.group_tag,))
    run_ids = [row[0] for row in cursor.fetchall()]
    if not run_ids:
        conn.close()
        print "No runs found with group_tag={} in {}".format(args.group_tag, db_path)
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
        SELECT run_tag, n_train, n_validation, sparsity_threshold
        FROM runs WHERE run_id = ?
    """, (run_id,))
    row = cursor.fetchone()
    if row is None:
        conn.close()
        print "run_id {} not found in {}".format(run_id, db_path)
        sys.exit(1)
    run_tag, n_train, n_validation, sparsity_threshold = row

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
        "epochs": epochs[plot_slice],
        "training_cost": training_cost[plot_slice],
        "training_accuracy": [float(a) / n_train for a in training_accuracy[plot_slice]],
        "evaluation_cost": evaluation_cost[plot_slice],
        "evaluation_accuracy": [float(a) / n_validation for a in evaluation_accuracy[plot_slice]],
        "weight_norm": weight_norm[plot_slice],
        "weight_sparsity": weight_sparsity[plot_slice],
    })

conn.close()

fig, ((ax_cost, ax_acc), (ax_norm, ax_sparsity)) = plt.subplots(2, 2, figsize=(10, 8))

for run in runs:
    label = "run{}".format(run["run_id"])
    line, = ax_cost.plot(run["epochs"], run["training_cost"], label="{} train".format(label))
    ax_cost.plot(run["epochs"], run["evaluation_cost"], linestyle="--", color=line.get_color(),
                 label="{} eval".format(label))
ax_cost.set_xlabel("Epoch")
ax_cost.set_ylabel("Cost")
ax_cost.set_title("Cost per epoch")
ax_cost.legend()

for run in runs:
    label = "run{}".format(run["run_id"])
    line, = ax_acc.plot(run["epochs"], run["training_accuracy"], label="{} train".format(label))
    ax_acc.plot(run["epochs"], run["evaluation_accuracy"], linestyle="--", color=line.get_color(),
                label="{} eval".format(label))
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_title("Accuracy per epoch")
ax_acc.legend()

for run in runs:
    ax_norm.plot(run["epochs"], run["weight_norm"], label="run{}".format(run["run_id"]))
ax_norm.set_xlabel("Epoch")
ax_norm.set_ylabel("Norm")
ax_norm.set_title("Weight norm per epoch")
ax_norm.legend()

for run in runs:
    ax_sparsity.plot(run["epochs"], run["weight_sparsity"], label="run{}".format(run["run_id"]))
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
