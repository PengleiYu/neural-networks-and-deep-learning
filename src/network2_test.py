import matplotlib
matplotlib.use("Agg")  # no display available in this environment
import matplotlib.pyplot as plt
import mnist_loader
import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.Network([784,30,10], cost = network2.CrossEntropyCost)

# SGD returns (evaluation_cost, evaluation_accuracy, training_cost, training_accuracy)
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(
    training_data, 30, 10, 0.5,
    lmbda = 5.0, evaluation_data = validation_data,
    monitor_evaluation_accuracy = True,
    monitor_evaluation_cost = True,
    monitor_training_accuracy = True,
    monitor_training_cost = True)

epochs = range(1, len(evaluation_cost) + 1)

print "Epoch\tTrainCost\tTrainAcc\tEvalCost\tEvalAcc"
for i, ep in enumerate(epochs):
    print "{}\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}".format(
        ep,
        training_cost[i],
        float(training_accuracy[i]) / len(training_data),
        evaluation_cost[i],
        float(evaluation_accuracy[i]) / len(validation_data))

fig, (ax_cost, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

ax_cost.plot(epochs, training_cost, label="Training cost")
ax_cost.plot(epochs, evaluation_cost, label="Evaluation cost")
ax_cost.set_xlabel("Epoch")
ax_cost.set_ylabel("Cost")
ax_cost.set_title("Cost per epoch")
ax_cost.legend()

ax_acc.plot(epochs, [float(a) / len(training_data) for a in training_accuracy], label="Training accuracy")
ax_acc.plot(epochs, [float(a) / len(validation_data) for a in evaluation_accuracy], label="Evaluation accuracy")
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy")
ax_acc.set_title("Accuracy per epoch")
ax_acc.legend()

plt.tight_layout()
plt.savefig("network2_test_result.png")

