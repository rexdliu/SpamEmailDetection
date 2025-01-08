import numpy as np
from Activations import sigmoid, relu, softmax
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights_hidden_output) + self.bias_output
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output, learning_rate):
        output_error = output - y
        output_delta = output_error * sigmoid(self.z2, derivative=True)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * relu(self.z1, derivative=True)

        self.weights_hidden_output -= self.a1.T.dot(output_delta) * learning_rate
        self.bias_output -= np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden -= X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden -= np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate


    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output,learning_rate)
            # Adjust learning rate by scaling the updates
            self.weights_hidden_output += learning_rate * self.weights_hidden_output
            self.bias_output += learning_rate * self.bias_output
            self.weights_input_hidden += learning_rate * self.weights_input_hidden
            self.bias_hidden += learning_rate * self.bias_hidden

            if (epoch+1) % 100 == 0:
                loss = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')

    def predict(self, X):
        output = self.forward(X)
        return np.round(output)