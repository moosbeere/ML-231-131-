import numpy as np

# arr1 = np.array([1,2,3,4,5])
# print(arr1)
# arr2 = np.array([[1,2,3], [4,5,6]])
# print(arr2)
# print(arr2.shape)
#
# arr_zero = np.zeros((3,3))
# print(arr_zero)
#
# arr_ones = np.ones((3,3))
# print(arr_ones)
#
# print(np.ones_like(arr_zero))
#
# random_ar = np.random.rand(2,3)
# print(random_ar)
#
# arr3 = np.array([1,2,3])
# arr4 = np.array([4,5,6])
#
# result = arr4 + arr3
# result = arr3 - arr4
# result = arr4 * arr3
# result = arr4 / arr3
# print(result)
#
# print(arr3[0])
# print(arr4[2])
# slice = arr3[1:3]
# print(slice)
#
# mask = arr3 > 1
# print(arr3[mask])
#
# print(arr4[arr4 < 5])
#
# mean = np.mean(arr3)
# print(mean)
#
# std = np.std(arr3)
# print(std)
#
# print(np.max(arr3))
# print(np.min(arr3))
#
# arr5 = np.array([[1,2],[3,4]])
# arr6 = np.array([[3,4],[5,6]])
#
# dot = arr5.dot(arr6)
# print(dot)
#
# arr7 = np.array([[2,3,1], [6,7,8]])
# print(arr7.T)

def neuralNetwork(inps, weights):
    prediction_h1 = inps.dot(weights[0])
    prediction_h2 = prediction_h1.dot(weights[1])
    prediction_out_h3 = prediction_h2 * weights[2]
    prediction_out = prediction_out_h3.dot(weights[3])
    return prediction_out

inp = np.array([23, 45])
weight_h_1 = np.array([0.4, 0.1])
weight_h_2 = np.array([0.3, 0.2])

weight_out_1 = np.array([0.4, 0.1])
weight_out_2 = np.array([0.3, 0.1])

weight_h1 = np.array([weight_h_1, weight_h_2]).T
weight_h2 = np.array([0.6, 0.2])
weight_out_h3 = np.array([0.7, 0.4])
weight_out = np.array([weight_out_1, weight_out_2]).T

weights = [weight_h1, weight_h2, weight_out_h3, weight_out]

print(neuralNetwork(inp, weights))

