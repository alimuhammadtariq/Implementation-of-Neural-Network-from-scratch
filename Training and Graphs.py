import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

# Load normalized training data and labels
X_train = pd.read_csv('x_train.csv').values
y_train = pd.read_csv('y_train.csv').values

# Load normalized testing data and labels
X_test = pd.read_csv('x_test.csv').values
y_test = pd.read_csv('y_test.csv').values

# Initialize neural network
input_size = X_train.shape[1]
hidden_size = 5
output_size = 2
eta = 0.001
nn = NeuralNetwork(input_size, hidden_size, output_size, eta)

# Training the neural network and collect training loss values
training_losses = []
testing_losses = []

def train_and_evaluate(X_train, y_train, X_test, y_test, epochs=1000):
    for epoch in range(epochs):
        predictions_train = nn.forward_pass(X_train)
        nn.backward_pass(X_train, y_train)

        if epoch % 100 == 0:
            # Calculate training loss
            training_loss = np.mean(np.square(predictions_train - y_train))
            training_losses.append(training_loss)

            # Make predictions on the testing data
            predictions_test = nn.forward_pass(X_test)

            # Calculate testing loss
            testing_loss = np.mean(np.square(predictions_test - y_test))
            testing_losses.append(testing_loss)
    nn.save_parameters()

    return training_losses, testing_losses

# Train and evaluate the network
training_losses, testing_losses = train_and_evaluate(X_train, y_train, X_test, y_test, epochs=10000)

# Plot the training and testing losses
epochs_range = range(0, 10000, 100)
plt.plot(epochs_range, training_losses, label='Training Loss')
plt.plot(epochs_range, testing_losses, label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
