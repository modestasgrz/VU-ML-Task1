from Neuron import Neuron
import numpy as np

# Training input values
input_vector = np.array([[-0.3, 0.6],
                         [0.3, -0.6],
                         [1.2, -1.2],
                         [1.2, 1.2]])

# Training output labels
label_vector = np.array([0, 0, 1, 1])

# Initialization of a neuron
neuron = Neuron(number_of_inputs = 2, activation="sigmoid")

# Training of a neuron
# Needed to find appropriate values of weights
neuron.train(epochs=20, learning_rate=30,
             training_data=input_vector,
             training_labels=label_vector)

# Output prediction
outputs = neuron.predict_outputs(input_vector)
print("Predicted outputs: {}".format(outputs))