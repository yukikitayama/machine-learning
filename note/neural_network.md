## Neural Network

### Big Picture

- Current model makes predictions
- Calculate loss
- Take gradients of the loss
- Update weights

### Loss Functions

- Hinge loss
    - Binary classification
    - Target values are expected to be -1 or 1.
    - In TensorFlow, if you provide 0 or 1 target value, TensorFlow automatically converts to -1 or 1.
    - Output layer should use hyperbolic tangent activation function `Dense(1, activation='tanh')` to convert a single
      value in the range [-1, 1].

### Weight Decay

- ESL Chapter 11
- Penalize by the sum-of-squares of the parameters like Ridge.

### Resource

- [How to Choose Loss Functions When Training Deep Learning Neural Networks](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)
