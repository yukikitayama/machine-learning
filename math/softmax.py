import numpy as np


def softmax(vector):
    e_vector = np.exp(vector)
    return e_vector / np.sum(e_vector)


probabilities = [0.0, 0.0, 0.1, 0.3, 0.9, 0.75, 0.1, 0.0, 0.0, 0.0]
print(probabilities)
print(sum(probabilities))

softmax_probabilities = softmax(probabilities)
print(np.round(softmax_probabilities, 2))
print(sum(softmax_probabilities))
print(sum([0.1, 0.19, 0.16, 0.08]))