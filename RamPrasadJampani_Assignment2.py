import numpy as np

# Define the sigmoid activation function
def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

# Define the ReLU activation function
def relu_activation(x):
    return np.maximum(0, x)

# Define the derivative of ReLU
def relu_derivative_func(x):
    return np.where(x > 0, 1, 0)

# Define the binary cross-entropy loss
def cross_entropy_loss(actual, predicted):
    return -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))

# Initialize weights and biases
np.random.seed(42)
hidden_weights = np.random.randn(2, 2)  # Weights for hidden layer
hidden_biases = np.random.randn(2)      # Biases for hidden layer
output_weights = np.random.randn(2, 1)  # Weights for output layer
output_bias = np.random.randn(1)        # Bias for output layer

# Input data (features) and labels
features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([[0], [1], [1], [0]])

# Hyperparameters
lr = 0.1
num_epochs = 10000

# Training loop
for epoch in range(num_epochs):
    # Forward propagation
    hidden_layer_input = np.dot(features, hidden_weights) + hidden_biases
    hidden_layer_output = relu_activation(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, output_weights) + output_bias
    predictions = sigmoid_activation(output_layer_input)

    # Compute loss
    loss = cross_entropy_loss(labels, predictions)

    # Backpropagation
    loss_gradient = (predictions - labels) / labels.shape[0]
    d_output = loss_gradient * predictions * (1 - predictions)
    d_output_weights = np.dot(hidden_layer_output.T, d_output)
    d_output_bias = np.sum(d_output, axis=0)
    d_hidden_output = np.dot(d_output, output_weights.T)
    d_hidden_input = d_hidden_output * relu_derivative_func(hidden_layer_input)
    d_hidden_weights = np.dot(features.T, d_hidden_input)
    d_hidden_biases = np.sum(d_hidden_input, axis=0)

    # Update weights and biases
    output_weights -= lr * d_output_weights
    output_bias -= lr * d_output_bias
    hidden_weights -= lr * d_hidden_weights
    hidden_biases -= lr * d_hidden_biases

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Test the network
hidden_layer_input = np.dot(features, hidden_weights) + hidden_biases
hidden_layer_output = relu_activation(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, output_weights) + output_bias
final_predictions = sigmoid_activation(output_layer_input)
print("Predictions:")
print(final_predictions)
