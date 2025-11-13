import numpy as np

def neural_networks(inps, weigths):
    return inps.dot(weigths)

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

inps = np.array([
    [150,40],
    [170,80],
    [160,90]
])

weights = np.array([0.2, 0.3])
true_predictions = np.array([50, 120, 140]);
learning_rate = 0.00001

for i in range(700):
    error = 0
    for j in range(len(inps)):
        current = inps[j]
        true_prediction = true_predictions[j]
        prediction = neural_networks(current, weights)
        error += get_error(true_prediction, prediction)
        print("Prediction: %.10f, Weights: %s, True_prediction: %.10f" % (prediction, weights, true_prediction))
        delta = (prediction - true_prediction) * current * learning_rate
        weights = weights - delta
        print("Errors: %.10f" % error)
    print("------------------")