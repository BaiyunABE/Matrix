import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class FeedforwardNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.bias_output = np.random.rand(output_size)

    def forward(self, x):
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward(self, x, y, learning_rate):
        output_error = y - self.final_output
        output_delta = output_error * sigmoid_derivative(self.final_output)
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += x.T.dot(hidden_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate


def train(network, x, y, epochs, learning_rate):
    for epoch in range(epochs):
        network.forward(x)
        network.backward(x, y, learning_rate)
        if epoch % 1000 == 0:
            loss = np.mean(np.square(y - network.final_output))
            print(f"Epoch {epoch} - Loss: {loss}")


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 二输入
y = np.array([[0], [1], [1], [0]])  # XOR门
network = FeedforwardNeuralNetwork(input_size=2, hidden_size=3, output_size=1)
train(network, x, y, epochs=10000, learning_rate=0.1)
print("Final Outputs after Training:")
print(network.forward(x))
