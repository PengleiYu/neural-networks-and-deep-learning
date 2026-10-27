"""toy_cost_demo.py
~~~~~~~~~~~~~~~~~~

Chapter 3's classic single-neuron demo: compare how fast QuadraticCost
and CrossEntropyCost learn when the neuron starts out "badly wrong".

One neuron, one fixed input x=1.0, desired output y=0.0. Weight and
bias are both initialized to 2.0, so the initial output is
sigmoid(2*1+2) = sigmoid(4) ~= 0.982 -- confidently wrong. The point
is to watch the cost curve for the first several dozen epochs: under
QuadraticCost the sigmoid is saturated (sigmoid_prime(4) is tiny), so
gradient descent barely moves at first; under CrossEntropyCost the
(a-y) delta has no sigmoid_prime factor, so it starts learning
immediately.
"""
from __future__ import division
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0 / (1.0 + __import__("numpy").exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

def run(cost_name, eta=0.15, n_epochs=300, w0=2.0, b0=2.0, x=1.0, y=0.0):
    w, b = w0, b0
    costs = []
    for epoch in range(n_epochs):
        z = w * x + b
        a = sigmoid(z)
        cost = 0.5 * (a - y) ** 2
        costs.append(cost)
        if cost_name == "quadratic":
            delta = (a - y) * sigmoid_prime(z)
        elif cost_name == "cross_entropy":
            delta = (a - y)
        else:
            raise ValueError(cost_name)
        w -= eta * delta * x
        b -= eta * delta
    return costs

if __name__ == "__main__":
    quadratic_costs = run("quadratic")
    cross_entropy_costs = run("cross_entropy")

    print "epoch\tquadratic_cost\tcross_entropy_cost"
    for i in range(0, len(quadratic_costs), 10):
        print "{}\t{:.6f}\t{:.6f}".format(i, quadratic_costs[i], cross_entropy_costs[i])

    plt.figure(figsize=(7, 5))
    plt.plot(quadratic_costs, label="QuadraticCost")
    plt.plot(cross_entropy_costs, label="CrossEntropyCost")
    plt.xlabel("epoch")
    plt.ylabel("cost (0.5*(a-y)^2, for comparability)")
    plt.title("Single neuron, badly-wrong start (w=b=2.0, target=0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("toy_cost_demo.png")
    print "Saved plot to toy_cost_demo.png"
