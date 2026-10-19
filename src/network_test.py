import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network

# simple test
# net = network.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


# test no hidden neuron

# for i in xrange(3):
#     print("no hidden neuron, cnt=%d" % i)
#     net = network.Network([784, 10])
#     net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# test hidden neuron count

# h_arr = [10,5,2,1]
# count=3
# for h in h_arr:
#     print("%d hidden neuron, %d training count" % (h,count))
#     for i in xrange(count):
#         print("training %d:" % (i))
#         net = network.Network([784, h, 10])
#         net.SGD(training_data, 30, 10, 3.0, test_data=test_data)



# test eta learn speed
eta_arr=[0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 100.0]
for eta in eta_arr:
    for cnt in xrange(3):
        print("eta = %f, cnt=%d" % (eta, cnt))
        net = network.Network([784, 30, 10])
        net.SGD(training_data, 30, 10, eta, test_data=test_data)

