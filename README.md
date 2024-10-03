# Neural Network Implementation

This repository contains the code for a basic neural network implemented from scratch using Python and NumPy. The network is designed for supervised learning tasks, using the backpropagation algorithm and gradient descent optimization.

## Features
- Forward and backward propagation with sigmoid activation functions.
- Weight and bias initialization.
- Support for training with momentum and regularization (optional).
- Save and load trained model parameters.
- Training and evaluation of neural networks on custom datasets.
- Plotting of training and testing loss for performance analysis.

## Installation

Clone the repository and install the required libraries:
```bash
git clone https://github.com/your-repo/neural-network.git
cd neural-network
pip install -r requirements.txt

## Usage

### Training the Neural Network
You can initialize and train the neural network using your dataset. The training data (X_train.csv) and labels (y_train.csv) must be in CSV format. Here's an example of how to train the neural network:
