# Neural Network Implementation

This repository contains the code for a basic neural network implemented from scratch using Python and NumPy. The network is designed for supervised learning tasks, using the backpropagation algorithm and gradient descent optimization.

## Features
- Forward and backward propagation with sigmoid activation functions.
- Weight and bias initialization.
- Support for training with momentum and regularization (optional).
- Save and load trained model parameters.
- Training and evaluation of neural networks on custom datasets.
- Plotting of training and testing loss for performance analysis.

## Installation and Usage
      ```bash
      git clone https://github.com/alimuhammadtariq/Reinforcement-learning-for-aerial-Navigation-.git
      cd Reinforcement-learning-for-aerial-Navigation
      pip install -r requirements.txt
      cd "Reinforcement-learning-for-aerial-Navigation-/Quadcopter Hovering"
      python "Quadcopter Hovering/PPO Predictions.py"



## Installation
Clone the repository and install the required libraries:
```bash
git clone https://github.com/your-repo/neural-network.git
cd neural-network
pip install -r requirements.txt

## Usage

## Training the Neural Network
You can initialize and train the neural network using your dataset. The training data (X_train.csv) and labels (y_train.csv) must be in CSV format. Here's an example of how to train the neural network:

    ```bash
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

## Predicting with the Trained Model
After training, you can use the trained model to make predictions on new data:
X_test = pd.read_csv('x_test.csv').values
predictions = nn.forward_pass(X_test)
