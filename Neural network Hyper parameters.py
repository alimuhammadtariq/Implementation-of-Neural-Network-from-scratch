import numpy as np

class NeuralNetworks:
    def __init__(self, input_size, hidden_size, output_size, eta):
        # Initialize weights and biases with small random values

        # input_size gets the no of column of input vector x for x.w because
        # (row_x*column_x) * (Row_W*column_W) or column_x=Row_W while column_W= no of nodes
        self.W_h = np.random.randn(input_size, hidden_size)
        self.b_h = np.zeros((1, hidden_size))
        # The output from first layer will be multipled to next weights. For valid Mul no of Colum of output from first layer
        #is feed as rows of second and output size is no of nodes we want in this case its final so output size=2 (x-vel, Y-vel)
        self.W_o = np.random.randn(hidden_size, output_size)
        self.b_o = np.zeros((1, output_size))
        self.eta=eta


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, X):
        # Forward pass through the network
        self.hidden_input = np.dot(X, self.W_h) + self.b_h  # h= X.W+b
        self.hidden_output = self.sigmoid(self.hidden_input)  # output from first layer after activation (h1,h2)
        self.output_layer_input = np.dot(self.hidden_output, self.W_o) + self.b_o # h.w+b hidden output (h1,h2) feed to next layer weights
        self.predictions = self.sigmoid(self.output_layer_input)

        return self.predictions
    def backward_pass(self, X, y, momentum=0.9, lmbda=0.01):
        # Compute gradients
        error_output = self.predictions - y
        delta_output = error_output * self.sigmoid_derivative(self.predictions)

        grad_W_o = np.dot(self.hidden_output.T, delta_output)
        grad_b_o = np.sum(delta_output, axis=0, keepdims=True)

        error_hidden = np.dot(delta_output, self.W_o.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)
        grad_W_h = np.dot(X.T, delta_hidden)
        grad_b_h = np.sum(delta_hidden, axis=0, keepdims=True)

        # Update weights and biases with momentum and regularization
        self.velocity_W_o = momentum * self.velocity_W_o - self.eta * (grad_W_o + lmbda * self.W_o)
        self.velocity_b_o = momentum * self.velocity_b_o - self.eta * grad_b_o
        self.velocity_W_h = momentum * self.velocity_W_h - self.eta * (grad_W_h + lmbda * self.W_h)
        self.velocity_b_h = momentum * self.velocity_b_h - self.eta * grad_b_h

        self.W_o += self.velocity_W_o
        self.b_o += self.velocity_b_o
        self.W_h += self.velocity_W_h
        self.b_h += self.velocity_b_h



    def train(self, X, y, epochs=1000, momentum=0.9, lmbda=0.01):
        self.velocity_W_o = np.zeros_like(self.W_o)
        self.velocity_b_o = np.zeros_like(self.b_o)
        self.velocity_W_h = np.zeros_like(self.W_h)
        self.velocity_b_h = np.zeros_like(self.b_h)

        for epoch in range(epochs):
            predictions = self.forward_pass(X)
            self.backward_pass(X, y, momentum, lmbda)

            if epoch % 100 == 0:
                loss = np.mean(np.square(predictions - y))
                print(f"Epoch {epoch}, Loss: {loss}")
        self.save_parameters()

# ... (rest of the code)
