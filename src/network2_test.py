import json
import matplotlib
matplotlib.use("Agg")  # no display available in this environment
import matplotlib.pyplot as plt
import mnist_loader
import network2
import random
import numpy as np

random.seed(12345678)
np.random.seed(12345678)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net_sizes = [784,30,10]
n_train = 50000
training_data = training_data[:n_train]
n_epochs = 30
mini_batch_size = 10
eta = 0.5
lmbda = 0.0

run_tag = "net_{}_ntrain{}_epochs{}_mb{}_eta{}_lmbda{}".format(
    "-".join(str(s) for s in net_sizes), n_train, n_epochs, mini_batch_size, eta, lmbda)

net = network2.Network(net_sizes, cost = network2.CrossEntropyCost)

# Set to an integer (e.g. 200) to only plot the last N epochs; None plots all epochs.
plot_last_n_epochs = None

monitor_evaluation_accuracy = True
monitor_evaluation_cost = True
monitor_training_accuracy = True
monitor_training_cost = True
monitor_weight_norm = True

# SGD2 returns (evaluation_cost, evaluation_accuracy, training_cost,
# training_accuracy, weight_norm). Any list whose corresponding monitor_*
# flag above is False comes back empty.
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy, weight_norm = net.SGD2(
    training_data, n_epochs, mini_batch_size, eta=eta,
    lmbda = lmbda, evaluation_data = validation_data,
    monitor_evaluation_accuracy = monitor_evaluation_accuracy,
    monitor_evaluation_cost = monitor_evaluation_cost,
    monitor_training_accuracy = monitor_training_accuracy,
    monitor_training_cost = monitor_training_cost,
    monitor_weight_norm = monitor_weight_norm,
    )

# Epoch count comes from n_epochs, not from any of the (possibly empty) result
# lists, since unmonitored metrics come back as empty lists.
epochs = range(1, n_epochs + 1)

series = [
    ("TrainCost", training_cost, monitor_training_cost),
    ("TrainAcc", training_accuracy, monitor_training_accuracy),
    ("EvalCost", evaluation_cost, monitor_evaluation_cost),
    ("EvalAcc", evaluation_accuracy, monitor_evaluation_accuracy),
    ("WeightNorm", weight_norm, monitor_weight_norm),
]
active_series = [(name, values) for name, values, monitored in series if monitored]

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

fig, ((ax_cost, ax_acc), (ax_norm, ax_unused)) = plt.subplots(2, 2, figsize=(10, 8))
ax_unused.axis("off")

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

plt.tight_layout()
plt.savefig("network2_test_result_{}.png".format(run_tag))

results = {
    "net_sizes": net_sizes,
    "n_train": n_train,
    "n_epochs": n_epochs,
    "mini_batch_size": mini_batch_size,
    "eta": eta,
    "lmbda": lmbda,
    "epochs": list(epochs),
    "evaluation_cost": evaluation_cost,
    "evaluation_accuracy": evaluation_accuracy,
    "training_cost": training_cost,
    "training_accuracy": training_accuracy,
    "weight_norm": weight_norm,
}
with open("network2_test_result_{}.json".format(run_tag), "w") as f:
    json.dump(results, f, indent=2)

