import argparse
import datetime
import itertools
import os
import sqlite3
import uuid
import mnist_loader
import network2
import random
import numpy as np

# random.seed(12345678)
# np.random.seed(12345678)

def parse_net_sizes_list(s):
    # Comma-separated layer sizes for one architecture, e.g. "784,30,10".
    # Use ';' to sweep multiple architectures, e.g. "784,30,10;784,100,10".
    return [[int(x) for x in arch.split(",")] for arch in s.split(";")]

def parse_int_list(s):
    return [int(x) for x in s.split(",")]

def parse_float_list(s):
    return [float(x) for x in s.split(",")]

parser = argparse.ArgumentParser()
parser.add_argument("--net_sizes", type=parse_net_sizes_list, default=[[784, 30, 10]],
                     help="Comma-separated layer sizes, e.g. 784,30,10; use ';' to sweep "
                          "multiple architectures, e.g. 784,30,10;784,100,10")
parser.add_argument("--n_train", type=parse_int_list, default=[1000],
                     help="Comma-separated list to sweep, e.g. 500,1000,2000")
parser.add_argument("--n_epochs", type=parse_int_list, default=[400])
parser.add_argument("--mini_batch_size", type=parse_int_list, default=[10])
parser.add_argument("--eta", type=parse_float_list, default=[0.5])
parser.add_argument("--lmbda", type=parse_float_list, default=[0.0],
                     help="L2 regularization parameter")
parser.add_argument("--lmbda1", type=parse_float_list, default=[0.0],
                     help="L1 regularization parameter")
parser.add_argument("--dropout_prob", type=parse_float_list, default=[0.0],
                     help="Probability of dropping a hidden-layer unit; 0.0 disables dropout")
parser.add_argument("--sparsity_threshold", type=parse_float_list, default=[1e-2])
parser.add_argument("--patience", type=parse_int_list, default=[0],
                     help="Early stopping patience (epochs with no evaluation-accuracy "
                          "improvement before stopping); 0 disables early stopping")
parser.add_argument("--group_tag", type=str, default=None,
                     help="Group tag prefix for this run; a unique suffix "
                          "(timestamp+random) is always appended, so repeated "
                          "values won't collide across invocations")
args = parser.parse_args()

# Every parameter above is a list (possibly of length 1); the Cartesian
# product of all of them is the set of (net_sizes, n_train, n_epochs, ...)
# combinations to run in this invocation. A plain single-value CLI call
# (the common case) produces exactly one combination, same as before.
combos = list(itertools.product(
    args.net_sizes, args.n_train, args.n_epochs, args.mini_batch_size,
    args.eta, args.lmbda, args.lmbda1, args.dropout_prob,
    args.patience, args.sparsity_threshold))

# One group_tag per script invocation, shared by every combination run this
# time, so they can later be pulled back out of the db as a batch. A unique
# suffix is always appended so a user-supplied --group_tag can't collide
# across separate invocations.
unique_suffix = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
group_tag = "{}-{}".format(args.group_tag, unique_suffix) if args.group_tag else unique_suffix

print "Running {} combination(s), group_tag={}".format(len(combos), group_tag)

full_training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
n_validation = len(validation_data)

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "network2_experiments.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at          TEXT NOT NULL,
        group_tag           TEXT,
        run_tag             TEXT NOT NULL,
        net_sizes           TEXT NOT NULL,
        n_train             INTEGER NOT NULL,
        n_validation        INTEGER NOT NULL,
        n_epochs_requested  INTEGER NOT NULL,
        actual_epochs       INTEGER NOT NULL,
        mini_batch_size     INTEGER NOT NULL,
        eta                 REAL NOT NULL,
        lmbda               REAL NOT NULL,
        lmbda1              REAL NOT NULL,
        dropout_prob        REAL NOT NULL,
        patience            INTEGER NOT NULL,
        sparsity_threshold  REAL NOT NULL
    )
""")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS epochs (
        run_id              INTEGER NOT NULL REFERENCES runs(run_id),
        epoch               INTEGER NOT NULL,
        training_cost       REAL,
        training_accuracy   INTEGER,
        evaluation_cost     REAL,
        evaluation_accuracy INTEGER,
        weight_norm         REAL,
        weight_sparsity     REAL,
        PRIMARY KEY (run_id, epoch)
    )
""")
# Older databases (created before group_tag existed) need a non-destructive
# migration: add the column, existing rows get NULL.
existing_columns = [row[1] for row in cursor.execute("PRAGMA table_info(runs)").fetchall()]
if "group_tag" not in existing_columns:
    cursor.execute("ALTER TABLE runs ADD COLUMN group_tag TEXT")
conn.commit()

for combo_index, (net_sizes, n_train, n_epochs, mini_batch_size, eta, lmbda,
                   lmbda1, dropout_prob, patience, sparsity_threshold) in enumerate(combos):

    run_tag = "net_{}_ntrain{}_epochs{}_mb{}_eta{}_lmbda_{}_lmbda1_{}_dropout_{}_patience_{}".format(
        "-".join(str(s) for s in net_sizes), n_train, n_epochs, mini_batch_size, eta, lmbda, lmbda1, dropout_prob, patience)

    print "=== combo {}/{}: {} ===".format(combo_index + 1, len(combos), run_tag)

    training_data = full_training_data[:n_train]

    net = network2.Network(net_sizes, cost = network2.CrossEntropyCost)

    monitor_evaluation_accuracy = True
    monitor_evaluation_cost = True
    monitor_training_accuracy = True
    monitor_training_cost = True
    monitor_weight_norm = True
    monitor_weight_sparsity = True

    # SGD2 returns (evaluation_cost, evaluation_accuracy, training_cost,
    # training_accuracy, weight_norm, weight_sparsity). Any list whose
    # corresponding monitor_* flag above is False comes back empty.
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy, weight_norm, weight_sparsity = net.SGD2(
        training_data, n_epochs, mini_batch_size, eta=eta,
        lmbda = lmbda, lmbda1 = lmbda1, dropout_prob = dropout_prob,
        evaluation_data = validation_data,
        monitor_evaluation_accuracy = monitor_evaluation_accuracy,
        monitor_evaluation_cost = monitor_evaluation_cost,
        monitor_training_accuracy = monitor_training_accuracy,
        monitor_training_cost = monitor_training_cost,
        monitor_weight_norm = monitor_weight_norm,
        monitor_weight_sparsity = monitor_weight_sparsity,
        sparsity_threshold = sparsity_threshold,
        patience = patience,
        )

    # Epoch count comes from the actual number of epochs run, not n_epochs,
    # since early stopping (patience > 0) can end training before n_epochs is
    # reached. monitor_evaluation_accuracy is hardcoded True above, so
    # evaluation_accuracy's length reliably reflects how many epochs ran.
    epochs = range(1, len(evaluation_accuracy) + 1)

    series = [
        ("TrainCost", training_cost, monitor_training_cost),
        ("TrainAcc", training_accuracy, monitor_training_accuracy),
        ("EvalCost", evaluation_cost, monitor_evaluation_cost),
        ("EvalAcc", evaluation_accuracy, monitor_evaluation_accuracy),
        ("WeightNorm", weight_norm, monitor_weight_norm),
        ("WeightSparsity", weight_sparsity, monitor_weight_sparsity),
    ]
    active_series = [(name, values) for name, values, monitored in series if monitored]

    print "net_sizes={} n_train={} n_epochs={} mini_batch_size={} eta={} lmbda={} lmbda1={} dropout_prob={} patience={} actual_epochs={}".format(
        net_sizes, n_train, n_epochs, mini_batch_size, eta, lmbda, lmbda1, dropout_prob, patience, len(epochs))
    print "Epoch\t" + "\t\t".join(name for name, _ in active_series)
    for i, ep in enumerate(epochs):
        columns = [str(ep)]
        for name, values in active_series:
            if name == "TrainAcc":
                columns.append("{:.4f}".format(float(values[i]) / len(training_data)))
            elif name == "EvalAcc":
                columns.append("{:.4f}".format(float(values[i]) / n_validation))
            elif name == "WeightNorm":
                columns.append("{:.4f}".format(values[i]))
            else:
                columns.append("{:.4f}".format(values[i]))
        print "\t\t".join(columns)

    cursor.execute("""
        INSERT INTO runs (
            created_at, group_tag, run_tag, net_sizes, n_train, n_validation,
            n_epochs_requested, actual_epochs, mini_batch_size, eta,
            lmbda, lmbda1, dropout_prob, patience, sparsity_threshold
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.datetime.utcnow().isoformat(), group_tag, run_tag,
        ",".join(str(s) for s in net_sizes), n_train, n_validation,
        n_epochs, len(epochs), mini_batch_size, eta,
        lmbda, lmbda1, dropout_prob, patience, sparsity_threshold,
    ))
    run_id = cursor.lastrowid

    cursor.executemany("""
        INSERT INTO epochs (
            run_id, epoch, training_cost, training_accuracy,
            evaluation_cost, evaluation_accuracy, weight_norm, weight_sparsity
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (run_id, ep, training_cost[i], training_accuracy[i],
         evaluation_cost[i], evaluation_accuracy[i], weight_norm[i], weight_sparsity[i])
        for i, ep in enumerate(epochs)
    ])

    conn.commit()
    print "Saved run_id={} (group_tag={}) to {}".format(run_id, group_tag, db_path)

conn.close()
print "Done. {} run(s) saved under group_tag={}".format(len(combos), group_tag)
