import argparse
import json
import matplotlib
matplotlib.use("Agg")  # no display available in this environment
import matplotlib.pyplot as plt
import mnist_loader
import network2
import random
import numpy as np

# random.seed(12345678)
# np.random.seed(12345678)

def parse_net_sizes(s):
    return [int(x) for x in s.split(",")]

parser = argparse.ArgumentParser()
parser.add_argument("--net_sizes", type=parse_net_sizes, default=[784, 30, 10],
                     help="Comma-separated layer sizes, e.g. 784,30,10")
parser.add_argument("--n_train", type=int, default=1000)
parser.add_argument("--n_epochs", type=int, default=400)
parser.add_argument("--mini_batch_size", type=int, default=10)
parser.add_argument("--eta", type=float, default=0.5)
parser.add_argument("--lmbda", type=float, default=0.0,
                     help="L2 regularization parameter")
parser.add_argument("--lmbda1", type=float, default=0.0,
                     help="L1 regularization parameter")
parser.add_argument("--dropout_prob", type=float, default=0.0,
                     help="Probability of dropping a hidden-layer unit; 0.0 disables dropout")
parser.add_argument("--plot_last_n_epochs", type=int, default=None,
                     help="Only plot the last N epochs; omit to plot all epochs")
parser.add_argument("--sparsity_threshold", type=float, default=1e-2)
parser.add_argument("--patience", type=int, default=0,
                     help="Early stopping patience (epochs with no evaluation-accuracy "
                          "improvement before stopping); 0 disables early stopping")
args = parser.parse_args()

net_sizes = args.net_sizes
n_train = args.n_train
n_epochs = args.n_epochs
mini_batch_size = args.mini_batch_size
eta = args.eta
lmbda = args.lmbda
lmbda1 = args.lmbda1
dropout_prob = args.dropout_prob
plot_last_n_epochs = args.plot_last_n_epochs
sparsity_threshold = args.sparsity_threshold
patience = args.patience

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = training_data[:n_train]

run_tag = "net_{}_ntrain{}_epochs{}_mb{}_eta{}_lmbda_{}_lmbda1_{}_dropout_{}_patience_{}".format(
    "-".join(str(s) for s in net_sizes), n_train, n_epochs, mini_batch_size, eta, lmbda, lmbda1, dropout_prob, patience)

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
            columns.append("{:.4f}".format(float(values[i]) / len(validation_data)))
        elif name == "WeightNorm":
            columns.append("{:.4f}".format(values[i]))
        else:
            columns.append("{:.4f}".format(values[i]))
    print "\t\t".join(columns)

if plot_last_n_epochs is None:
    plot_slice = slice(0, len(epochs))
else:
    plot_slice = slice(max(0, len(epochs) - plot_last_n_epochs), len(epochs))

plot_epochs = epochs[plot_slice]

fig, ((ax_cost, ax_acc), (ax_norm, ax_sparsity)) = plt.subplots(2, 2, figsize=(10, 8))

if monitor_training_cost:
    ax_cost.plot(plot_epochs, training_cost[plot_slice], label="Training cost")
if monitor_evaluation_cost:
    ax_cost.plot(plot_epochs, evaluation_cost[plot_slice], label="Evaluation cost")
ax_cost.set_xlabel("Epoch")
ax_cost.set_ylabel("Cost")
ax_cost.set_title("Cost per epoch")
if monitor_training_cost or monitor_evaluation_cost:
    ax_cost.legend()

if monitor_training_accuracy:
    ax_acc.plot(plot_epochs,
                [float(a) / len(training_data) for a in training_accuracy[plot_slice]],
                label="Training accuracy")
if monitor_evaluation_accuracy:
    ax_acc.plot(plot_epochs,
                [float(a) / len(validation_data) for a in evaluation_accuracy[plot_slice]],
                label="Evaluation accuracy")
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_title("Accuracy per epoch")
if monitor_training_accuracy or monitor_evaluation_accuracy:
    ax_acc.legend()

if monitor_weight_norm:
    ax_norm.plot(plot_epochs, weight_norm[plot_slice], label="Weight norm")
ax_norm.set_xlabel("Epoch")
ax_norm.set_ylabel("Norm")
ax_norm.set_title("Weight norm per epoch")
if monitor_weight_norm:
    ax_norm.legend()

if monitor_weight_sparsity:
    ax_sparsity.plot(plot_epochs, weight_sparsity[plot_slice], label="Weight sparsity")
ax_sparsity.set_xlabel("Epoch")
ax_sparsity.set_ylabel("Fraction |w| < {}".format(sparsity_threshold))
ax_sparsity.set_title("Weight sparsity per epoch")
if monitor_weight_sparsity:
    ax_sparsity.legend()

plt.tight_layout()
plt.savefig("network2_test_result_{}.png".format(run_tag))

results = {
    "net_sizes": net_sizes,
    "n_train": n_train,
    "n_epochs": n_epochs,
    "mini_batch_size": mini_batch_size,
    "eta": eta,
    "lmbda": lmbda,
    "lmbda1": lmbda1,
    "dropout_prob": dropout_prob,
    "patience": patience,
    "actual_epochs": len(epochs),
    "evaluation_cost": evaluation_cost,
    "evaluation_accuracy": evaluation_accuracy,
    "training_cost": training_cost,
    "training_accuracy": training_accuracy,
    "weight_norm": weight_norm,
    "weight_sparsity": weight_sparsity,
    "sparsity_threshold": sparsity_threshold,
}
with open("network2_test_result_{}.json".format(run_tag), "w") as f:
    json.dump(results, f, indent=2)

