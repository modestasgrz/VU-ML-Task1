import numpy as np
import math


# Perceptron model
class Neuron(object):
    # Constructor of the perceptron which takes the type of activation function, which will be used in perceptron,
    # and number of inputs - to initialize appropriate number of weights
    def __init__(self, number_of_inputs, activation="step"):
        self.number_of_inputs = number_of_inputs
        self.weights_vector = np.random.rand(number_of_inputs + 1)
        self.activation = activation

    # Function that implements step calculation
    def step(self, result):
        if result < 0:
            return 0
        else:
            return 1

    # Function that implements sigmoid calculation
    def sigmoid(self, result):
        hypothesis = 1 / (1 + math.exp(-result))
        return hypothesis

    # Function that calculates the cost function of the current prediction and input values
    def calculate_cost_function(self, predicted_outputs, training_labels):
        cost_function = sum(
            (-training_labels * np.log(predicted_outputs)) -
            ((1 - training_labels) * np.log(1 - predicted_outputs))
        ) / len(training_labels)
        return cost_function

    # Function needed to train the perceptron
    # Find optimal weight values by changing them according to gradient decent
    def train(self, epochs, learning_rate, training_data, training_labels):
        # Adds additional bias column of ones to the input matrix
        training_data = np.c_[np.ones(len(training_data)), training_data]
        for i in range(epochs):
            # Calculates predicted outputs of hypothesis
            predicted_outputs = self.calculate_outputs(training_data)

            # Cost function calculation, len(training_data[0]) - number of training examples
            cost_function = self.calculate_cost_function(predicted_outputs, training_labels)
            print("Epoch: {}".format(i + 1))
            print("Cost function value: {}".format(cost_function))

            # Implement gradient decent step. Vectorized implementation.
            gradient = ((predicted_outputs - training_labels) @ training_data) / len(training_labels)

            # Recalculate weights according to gradient decent
            self.weights_vector = self.weights_vector - learning_rate * gradient

            print("Current weights' values: {}".format(self.weights_vector))

    # Function that calculates predicted outputs according to current values of the weights
    def calculate_outputs(self, input_data):
        # Vectorized implementation of output calculation
        calculated_products = input_data @ self.weights_vector.T

        # Choose appropriate method to determine output values regarding specified activation function
        if self.activation == "step":
            return np.array([self.step(y) for y in calculated_products])
        elif self.activation == "sigmoid":
            return np.array([self.sigmoid(y) for y in calculated_products])
        else:
            # Raise exception if specified function is not implemented or non-existent
            raise Exception("The specified activation function is not implemented or non-existent")

    # Function that predicts outputs according to inputs using current weights' values
    def predict_outputs(self, input_data):
        if len(input_data[0]) != self.number_of_inputs:
            raise Exception("Number of input elements is not equal to specified number")
        # Adds additional bias column of ones to the input matrix
        input_data = np.c_[np.ones(len(input_data)), input_data]
        outputs = self.calculate_outputs(input_data)
        return np.array([round(y) for y in outputs])

