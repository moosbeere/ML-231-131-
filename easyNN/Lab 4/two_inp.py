import numpy as np

def neural_networks(inps, weigths):
    return inps.dot(weigths)

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

inps = np.array([150,40])
weights = np.array([0.2, 0.3])

true_prediction = 1
learning_rate = 0.0001

for i in range(158):
    prediction = neural_networks(inps, weights)
    error = get_error(true_prediction, prediction)
    print("Prediction: %.10f, Weights: %s, Error: %.20f" % (prediction, weights, error))
    delta = (prediction - true_prediction) * inps * learning_rate
    delta[0] = 0
    weights = weights - delta