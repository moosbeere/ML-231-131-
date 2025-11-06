import numpy as np

def neuralNetwork(inps, weights):
    prediction = inps * weights
    return prediction

def get_error(true_prediction, prediction):
    return (true_prediction - prediction)**2

inp = 0.9
weight = 0.2
true_prediction = 0.2

for i in range(13):
    prediction = neuralNetwork(inp, weight)
    error = get_error(true_prediction,prediction)
    print("Prediction: %.10f, Weight: %.5f, Error: %.20f" % (prediction, weight, error))
    delta = (prediction - true_prediction) * inp
    weight -= delta;
