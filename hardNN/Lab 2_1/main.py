import numpy as np

def relu(x):
    for i in range(len(x)):
        if (x[i] < 0): x[i] = 0
    return x

inp = np.array([
    [15,10],
    [15,15],
    [15,20],
    [25,10]
])
true_prediction = np.array([10,20,15,20])
# true_prediction = np.array([[10, 20, 15, 20]]).T

layer_hid_size = 3
layer_in_size = len(inp[0])
# layer_out_size = len(true_prediction[0])
layer_out_size = 1

weight_hid = 2 * np.random.random((layer_in_size, layer_hid_size)) - 1
weight_out = np.random.random((layer_hid_size, layer_out_size))

print(weight_hid)
# print(weight_out)

prediction_hid = relu(inp[0].dot(weight_hid))
print(prediction_hid)
prediction_out = prediction_hid.dot(weight_out)
print(prediction_out)
