import numpy as np
from Activations import sigmoid, relu, softmax
def binary_cross_entropy(y_true, y_pred, eps=1e-9):
    return -np.mean(
        y_true * np.log(y_pred + eps) +
        (1 - y_true) * np.log(1 - y_pred + eps)
    )
    def mixed_loss(y_true, y_pred, alpha=0.5, beta=0.5, eps=1e-9):
    y_true_cls, y_true_reg = y_true[:, 0], y_true[:, 1]
    y_pred_cls, y_pred_reg = y_pred[:, 0], y_pred[:, 1]
    y_pred_cls = np.clip(y_pred_cls, eps, 1 - eps)
    bce_loss = -np.mean(
        y_true_cls * np.log(y_pred_cls) +
        (1 - y_true_cls) * np.log(1 - y_pred_cls)
    )
    mse_loss = np.mean((y_true_reg - y_pred_reg) ** 2)
    return alpha * bce_loss + beta * mse_loss

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def categorical_cross_entropy(y_true, y_pred, eps=1e-9):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

        # Adam parameters
        self.learning_rate = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m_wih = np.zeros_like(self.weights_input_hidden)
        self.v_wih = np.zeros_like(self.weights_input_hidden)
        self.m_who = np.zeros_like(self.weights_hidden_output)
        self.v_who = np.zeros_like(self.weights_hidden_output)
        self.m_bh = np.zeros_like(self.bias_hidden)
        self.v_bh = np.zeros_like(self.bias_hidden)
        self.m_bo = np.zeros_like(self.bias_output)
        self.v_bo = np.zeros_like(self.bias_output)
        self.iterations = 0

    def forward(self, X):
        self.z1 = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.a1 = np.maximum(self.z1, 0)  # ReLU activation
        self.z2 = np.dot(self.a1, self.weights_hidden_output) + self.bias_output
        self.a2 = 1 / (1 + np.exp(-self.z2))  # Sigmoid activation
        return self.a2

    def backward(self, X, y, output):
        # Output layer error
        output_error = output - y
        output_delta = output_error * output * (1 - output)  # Sigmoid derivative

        # Hidden layer error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * (self.z1 > 0).astype(float)  # ReLU derivative

        # Gradients for weights and biases
        grad_wih = X.T.dot(hidden_delta)
        grad_who = self.a1.T.dot(output_delta)
        grad_bh = np.sum(hidden_delta, axis=0, keepdims=True)
        grad_bo = np.sum(output_delta, axis=0, keepdims=True)

        # Update moment vectors and weights for input-hidden layer
        self.m_wih = self.beta1 * self.m_wih + (1 - self.beta1) * grad_wih
        self.v_wih = self.beta2 * self.v_wih + (1 - self.beta2) * (grad_wih ** 2)
        m_hat_wih = self.m_wih / (1 - self.beta1 ** (self.iterations + 1))
        v_hat_wih = self.v_wih / (1 - self.beta2 ** (self.iterations + 1))
        self.weights_input_hidden -= self.learning_rate * m_hat_wih / (np.sqrt(v_hat_wih) + self.epsilon)

        # Update moment vectors and weights for hidden-output layer
        self.m_who = self.beta1 * self.m_who + (1 - self.beta1) * grad_who
        self.v_who = self.beta2 * self.v_who + (1 - self.beta2) * (grad_who ** 2)
        m_hat_who = self.m_who / (1 - self.beta1 ** (self.iterations + 1))
        v_hat_who = self.v_who / (1 - self.beta2 ** (self.iterations + 1))
        self.weights_hidden_output -= self.learning_rate * m_hat_who / (np.sqrt(v_hat_who) + self.epsilon)

        # Update biases using Adam
        self.m_bh = self.beta1 * self.m_bh + (1 - self.beta1) * grad_bh
        self.v_bh = self.beta2 * self.v_bh + (1 - self.beta2) * (grad_bh ** 2)
        m_hat_bh = self.m_bh / (1 - self.beta1 ** (self.iterations + 1))
        v_hat_bh = self.v_bh / (1 - self.beta2 ** (self.iterations + 1))
        self.bias_hidden -= self.learning_rate * m_hat_bh / (np.sqrt(v_hat_bh) + self.epsilon)

        self.m_bo = self.beta1 * self.m_bo + (1 - self.beta1) * grad_bo
        self.v_bo = self.beta2 * self.v_bo + (1 - self.beta2) * (grad_bo ** 2)
        m_hat_bo = self.m_bo / (1 - self.beta1 ** (self.iterations + 1))
        v_hat_bo = self.v_bo / (1 - self.beta2 ** (self.iterations + 1))
        self.bias_output -= self.learning_rate * m_hat_bo / (np.sqrt(v_hat_bo) + self.epsilon)

        # Increment the iteration count
        self.iterations += 1
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)


            if (epoch+1) % 100 == 0:
                loss = binary_cross_entropy(y, output)
                print(f'Epochs {epoch+1}/{epochs}, Loss: {loss:.4f}')

    def predict(self, X):
        output = self.forward(X)
        return np.round(output)
