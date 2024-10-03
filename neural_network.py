import numpy as np

class NeuralNetwork:
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


#Sigmoid Function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Sigmoid Function Derivative
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, X):
        # Forward pass through the network
        self.hidden_input = np.dot(X, self.W_h) + self.b_h  # h= X.W+b
        self.hidden_output = self.sigmoid(self.hidden_input)  # output from first layer after activation (h1,h2)
        self.output_layer_input = np.dot(self.hidden_output, self.W_o) + self.b_o # h.w+b hidden output (h1,h2) feed to next layer weights
        self.predictions = self.sigmoid(self.output_layer_input) # output from final layer or pridictions

        return self.predictions

#Backpropogation
    def backward_pass(self, X, y):
        # Compute gradients
        error_output =  self.predictions - y #calucating error from final layer y"1-Y1
        delta_output = error_output * self.sigmoid_derivative(self.predictions) # λ(y-(1-y))error  Local gradient in final layer
        grad_W_o = np.dot(self.hidden_output.T, delta_output) #local-gradient*h to get weight update  matrix we need to transpose the h for coreect size of weight Matrix
        grad_b_o = np.sum(delta_output, axis=0, keepdims=True)

        error_hidden = np.dot(delta_output, self.W_o.T) # derivative from chain rule (local_gradied*w) sigma local gradient in final layer*delta w in final. The last sum term in caluclating local gradient in hiddenlayer
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output) #finally caluating local gradient in hidden layer
        grad_W_h = np.dot(X.T, delta_hidden) #delta weight of hidden layer
        grad_b_h = np.sum(delta_hidden, axis=0, keepdims=True)

        # Update weights and biases
        self.W_o -= self.eta * grad_W_o
        self.b_o -= self.eta * grad_b_o
        self.W_h -= self.eta * grad_W_h
        self.b_h -= self.eta * grad_b_h

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            predictions = self.forward_pass(X)
            self.backward_pass(X, y)

            if epoch % 100 == 0:   #for ever 100 epoch print loss
                loss = np.mean(np.square(predictions - y))
                print(f"Epoch {epoch}, Loss: {loss}")
        self.save_parameters()  #save the weights

    def save_parameters(self):
        np.savetxt('trained_parameters.txt', np.concatenate([self.W_h.flatten(), self.b_h.flatten(), self.W_o.flatten(), self.b_o.flatten()]), delimiter=',')

    def load_parameters(self):
        parameters = np.loadtxt('trained_parameters.txt', delimiter=',')
        start_idx = 0

        # Update weights and biases with loaded parameters
        self.W_h = parameters[start_idx:start_idx + self.W_h.size].reshape(self.W_h.shape)
        start_idx += self.W_h.size

        self.b_h = parameters[start_idx:start_idx + self.b_h.size].reshape(self.b_h.shape)
        start_idx += self.b_h.size

        self.W_o = parameters[start_idx:start_idx + self.W_o.size].reshape(self.W_o.shape)
        start_idx += self.W_o.size

        self.b_o = parameters[start_idx:start_idx + self.b_o.size].reshape(self.b_o.shape)

# Example usage
# Assume X_train and y_train are your training data and labels
# input_size = X_train.shape[1] #No of columns in Input
# hidden_size = 4  # You can adjust this based on your problem
# output_size = 1
# nn = NeuralNetwork(input_size, hidden_size, output_size, eta)
#
# # Training the neural network
# nn.train(X_train, y_train, epochs=1000)

# After training, you can use nn.forward_pass(X_test) to get predictions for new data.
