# Implementation-of-Neural-Network-from-scratch

Designed and implemented a fully connected neural network from scratch to predict object velocities (X and Y) based on positional data. The network architecture included an input layer, a hidden layer, and an output layer, using the sigmoid activation function. The project involved training the network using backpropagation with momentum and regularization techniques, managing data preprocessing using MinMaxScaler, and evaluating model performance with loss tracking over epochs. The trained model was then deployed to make predictions on new data inputs, with real-time velocity predictions denormalized to their original scale.
Neural Network Implementation
This repository contains the code for a basic neural network implemented from scratch using Python and NumPy. The network is designed for supervised learning tasks, using the backpropagation algorithm and gradient descent optimization.

Features
Forward and backward propagation with sigmoid activation functions.
Weight and bias initialization.
Support for training with momentum and regularization (optional).
Save and load trained model parameters.
Training and evaluation of neural networks on custom datasets.
Plotting of training and testing loss for performance analysis.
Installation
Clone the repository and install the required libraries:

bash
Copy code
git clone https://github.com/your-repo/neural-network.git
cd neural-network
pip install -r requirements.txt
Usage
Training the Neural Network
You can initialize and train the neural network using your dataset. The training data (X_train.csv) and labels (y_train.csv) must be in CSV format. Here's an example of how to train the neural network:

python
Copy code
from neural_network import NeuralNetwork
import pandas as pd

# Load training data
X_train = pd.read_csv('x_train.csv').values
y_train = pd.read_csv('y_train.csv').values

# Initialize neural network
input_size = X_train.shape[1]
hidden_size = 5  # Adjust based on your problem
output_size = 2  # Number of output classes
eta = 0.001      # Learning rate

nn = NeuralNetwork(input_size, hidden_size, output_size, eta)

# Train the neural network
nn.train(X_train, y_train, epochs=1000)
Predicting with the Trained Model
After training, you can use the trained model to make predictions on new data:

python
Copy code
X_test = pd.read_csv('x_test.csv').values
predictions = nn.forward_pass(X_test)
Saving and Loading Model Parameters
The model parameters (weights and biases) are saved to a text file after training. You can load the saved parameters for future use.

Save Parameters
python
Copy code
nn.save_parameters()
Load Parameters
python
Copy code
nn.load_parameters()
Data Preprocessing
Use the preprocessing.py script to normalize and split your dataset into training, validation, and testing sets. The script also saves the scaling objects for future use.

bash
Copy code
python preprocessing.py
Visualizing Training Progress
The script also includes a function to plot the training and testing loss during training:

python
Copy code
import matplotlib.pyplot as plt

plt.plot(epochs_range, training_losses, label='Training Loss')
plt.plot(epochs_range, testing_losses, label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
Requirements
Python 3.x
NumPy
Pandas
Matplotlib
scikit-learn
Install the required packages with:

bash
Copy code
pip install numpy pandas matplotlib scikit-learn joblib
File Structure
neural_network.py: Contains the implementation of the neural network class.
preprocessing.py: Preprocesses the dataset by scaling and splitting it into training, validation, and testing sets.
train.py: Trains the neural network using the processed data.
x_train.csv, y_train.csv, x_test.csv, y_test.csv: Example datasets for training and testing.
