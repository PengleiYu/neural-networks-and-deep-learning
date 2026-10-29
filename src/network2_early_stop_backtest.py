from __future__ import division
import argparse
import os
import sqlite3

def parse_int_list(s):
    return [int(x) for x in s.split(",")]

parser = argparse.ArgumentParser()
parser.add_argument("--db", type=str, default=None,
                     help="Path to the SQLite db; defaults to network2_experiments.db next to this script")
parser.add_argument("--group_tag_prefix", type=str, default="exp_",
                     help="Only backtest runs whose group_tag starts with this prefix")
parser.add_argument("--patience_values", type=parse_int_list, default=[5, 10, 20],
                     help="Comma-separated patience values to simulate")
parser.add_argument("--group_by", choices=["group_tag", "net_sizes"], default="group_tag",
                     help="How to bucket the per-group breakdown: by experiment group_tag "
                          "(default, i.e. by purpose -- a tag may mix several architectures) "
                          "or by net_sizes (i.e. by actual architecture, regardless of which "
                          "experiment/regularization it came from)")
args = parser.parse_args()

db_path = args.db
if db_path is None:
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "network2_experiments.db")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Only full curves (patience=0, i.e. no early stopping was applied when the run
# was generated) can be backtested -- a run that already stopped early is
# missing the "what would have happened if we kept training" epochs.
cursor.execute("""
    SELECT run_id, group_tag, run_tag, actual_epochs, n_validation, net_sizes
    FROM runs
    WHERE group_tag LIKE ? AND patience = 0
    ORDER BY run_id
""", (args.group_tag_prefix + "%",))
runs = cursor.fetchall()

if not runs:
    print "No patience=0 runs found with group_tag prefix {}".format(args.group_tag_prefix)
    conn.close()
    raise SystemExit(1)

def logical_group(group_tag):
    # Strip the "-<timestamp>-<random>" suffix network2_test.py always appends,
    # so runs from a 16-way-parallel sweep (16 different literal group_tag
    # strings) still get pooled under one label.
    parts = group_tag.split("-")
    # suffix is always exactly 2 trailing "-"-joined tokens: YYYYMMDDTHHMMSS and a hex string
    if len(parts) >= 3:
        return "-".join(parts[:-2])
    return group_tag

def simulate_early_stop(evaluation_accuracy, patience):
    best_accuracy = 0
    best_index = -1
    no_improvement_count = 0
    for i, accuracy in enumerate(evaluation_accuracy):
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_index = i
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if no_improvement_count >= patience:
            return best_accuracy, best_index, i
    return best_accuracy, best_index, len(evaluation_accuracy) - 1

per_patience_results = dict((p, []) for p in args.patience_values)
per_group_per_patience = {}

for run_id, group_tag, run_tag, actual_epochs, n_validation, net_sizes in runs:
    cursor.execute("""
        SELECT evaluation_accuracy FROM epochs WHERE run_id = ? ORDER BY epoch
    """, (run_id,))
    evaluation_accuracy = [row[0] for row in cursor.fetchall()]
    if len(evaluation_accuracy) < 2:
        continue

    full_schedule_accuracy = evaluation_accuracy[-1]
    best_ever_accuracy = max(evaluation_accuracy)
    group_label = net_sizes if args.group_by == "net_sizes" else logical_group(group_tag)

    # Raw counts -> percentage points, using this run's own validation set size.
    to_pp = lambda raw_count_diff: 100.0 * raw_count_diff / n_validation

    for patience in args.patience_values:
        best_accuracy, best_index, stop_index = simulate_early_stop(evaluation_accuracy, patience)
        epochs_used = stop_index + 1
        gap_vs_full = to_pp(best_accuracy - full_schedule_accuracy)
        optimism_bias = to_pp(best_ever_accuracy - best_accuracy)
        epochs_saved_frac = 1.0 - epochs_used / len(evaluation_accuracy)

        record = {
            "run_id": run_id,
            "run_tag": run_tag,
            "gap_vs_full": gap_vs_full,
            "optimism_bias": optimism_bias,
            "epochs_saved_frac": epochs_saved_frac,
            "epochs_used": epochs_used,
            "total_epochs": len(evaluation_accuracy),
        }
        per_patience_results[patience].append(record)
        per_group_per_patience.setdefault(group_label, {}).setdefault(patience, []).append(record)

conn.close()

print "Backtest over {} full curves (group_tag_prefix={})".format(len(runs), args.group_tag_prefix)
print ""

for patience in args.patience_values:
    records = per_patience_results[patience]
    n = len(records)
    if n == 0:
        continue
    n_worse = sum(1 for r in records if r["gap_vs_full"] < 0)
    n_better = sum(1 for r in records if r["gap_vs_full"] > 0)
    mean_gap = sum(r["gap_vs_full"] for r in records) / n
    worst_gap = min(r["gap_vs_full"] for r in records)
    best_gap = max(r["gap_vs_full"] for r in records)
    mean_saved = 100.0 * sum(r["epochs_saved_frac"] for r in records) / n
    mean_optimism = sum(r["optimism_bias"] for r in records) / n
    worst_optimism = max(r["optimism_bias"] for r in records)

    print "patience={}".format(patience)
    print "  curves where naive full-schedule run would have been WORSE than the early-stop checkpoint: {}/{} ({:.1f}%)".format(
        n_worse, n, 100.0 * n_worse / n)
    print "  curves where full-schedule run would have been BETTER (early stop missed a later true peak): {}/{} ({:.1f}%)".format(
        n_better, n, 100.0 * n_better / n)
    print "  mean gap (early-stop best vs full-schedule final), pct points: {:+.2f}".format(mean_gap)
    print "  worst case for early stopping (most negative gap), pct points: {:+.2f}".format(worst_gap)
    print "  best case for early stopping (most positive gap), pct points: {:+.2f}".format(best_gap)
    print "  mean epochs saved: {:.1f}%".format(mean_saved)
    print "  'all-time-high' optimism bias (best_ever - early_stop_best), mean/worst pct points: {:.2f} / {:.2f}".format(
        mean_optimism, worst_optimism)
    print ""

print "Breakdown by experiment group:"
for group_label in sorted(per_group_per_patience.keys()):
    print "  {}:".format(group_label)
    for patience in args.patience_values:
        records = per_group_per_patience[group_label].get(patience, [])
        if not records:
            continue
        n = len(records)
        n_worse = sum(1 for r in records if r["gap_vs_full"] < 0)
        mean_gap = sum(r["gap_vs_full"] for r in records) / n
        print "    patience={}: n={}, full-schedule-worse={}/{} ({:.1f}%), mean_gap={:+.2f}pp".format(
            patience, n, n_worse, n, 100.0 * n_worse / n, mean_gap)
