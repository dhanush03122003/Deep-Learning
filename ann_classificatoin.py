import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

Y = iris.target_names[iris.target]

X = iris_df.drop('sepal length (cm)', axis=1)

x_train, x_test, y_train, y_test = map(np.array, train_test_split(X, Y, test_size=0.2, random_state=42))
y_train = np.array(pd.get_dummies(y_train))

l_rate = 0.01

echo  = 10000 # iterations

input_size = len(x_train[0])
hidden_lay = [8]
output_size = len(set(Y))

def sigmoid(p):
     return 1 / (1 + np.exp(-p))

lb = -1
ub = 1

def initialize_weights(input_size, hidden_layers, output_size, lb, ub):
    weight_mat = []

    # Initialize weights for the first layer
    w_first = np.random.randn(input_size, hidden_layers[0]) * (ub - lb) + lb
    weight_mat.append(w_first)

    # Initialize weights for the hidden layers
    for i in range(len(hidden_layers) - 1):
        wei = np.random.randn(hidden_layers[i], hidden_layers[i+1]) * (ub - lb) + lb
        weight_mat.append(wei)

    # Initialize weights for the last layer
    w_last = np.random.randn(hidden_layers[-1], output_size) * (ub - lb) + lb
    weight_mat.append(w_last)

    return weight_mat

def initialize_biases(hidden_layers, output_nodes, lb, ub):
    bias_mat = []

    # Initialize biases for the hidden layers
    for hidden_nodes in hidden_layers:
        b = np.random.rand(hidden_nodes) * (ub - lb) + lb
        bias_mat.append(b)

    # Initialize biases for the output layer
    b = np.random.rand(output_nodes) * (ub - lb) + lb
    bias_mat.append(b)

    return bias_mat

def forward_propagation(x_train, weight_matrices, bias_matrices):
    values = []
    for i, weight_mat in enumerate(weight_matrices):
        if i == 0:
            v = np.dot(x_train, weight_mat) + bias_matrices[i]
        else:
            v = np.dot(values[-1], weight_mat) + bias_matrices[i]
        v = np.array([sigmoid(val) for val in v])  # Apply sigmoid activation function
        values.append(v)
    return values

def backward_propagation(y_train, values, weight_matrices):
    errors = []
    error = (values[-1] * (1 - values[-1])) * (y_train - values[-1])
    errors.append(error)

    for i in range(len(hidden_lay)-1, -1, -1):
        h = (values[i] * (1 - values[i])) * (errors[-1] @ weight_matrices[i+1].T)
        errors.append(h)
    errors.reverse()
    return errors

def update_weights(x_train, values, errors, weight_matrices, bias_matrices, l_rate):
    for i, weight_mat in enumerate(weight_matrices):
        if i == 0:
            del_w = l_rate * (x_train.T @ errors[i])
        else:
            del_w = l_rate * (values[i-1].T @ errors[i])
        weight_matrices[i] += del_w

    for i, bias_mat in enumerate(bias_matrices):
        bias_matrices[i] += l_rate * np.sum(errors[i], axis=0)

def train_neural_network(x_train, y_train, weight_matrices, bias_matrices, l_rate, epochs):
    for _ in range(epochs):
        # Forward propagation
        values = forward_propagation(x_train, weight_matrices, bias_matrices)

        # Backward propagation
        errors = backward_propagation(y_train, values, weight_matrices)

        # Update weights and biases
        update_weights(x_train, values, errors, weight_matrices, bias_matrices, l_rate)

# Function for forward propagation on new data
def forward_propagation_new_data(new_x, weight_matrices, bias_matrices):
    predictions = []
    for i, weight_mat in enumerate(weight_matrices):
        if i == 0:
            v = np.dot(new_x, weight_mat) + bias_matrices[i]
        else:
            v = np.dot(predictions[-1], weight_mat) + bias_matrices[i]
        v = np.array([sigmoid(val) for val in v])  # Apply sigmoid activation function
        predictions.append(v)
    return predictions

# Function for predicting classes
def predict_classes(new_x, weight_matrices, bias_matrices, threshold=0.5):
    predictions = forward_propagation_new_data(new_x, weight_matrices, bias_matrices)
    final_predictions = (predictions[-1] >= threshold).astype(int)
    return final_predictions

weight_mat = initialize_weights(input_size, hidden_lay, output_size, lb, ub)
bias_mat = initialize_biases(hidden_lay, output_size, lb, ub)

train_neural_network(x_train, y_train, weight_mat, bias_mat, l_rate, echo)

# Assuming weight_matrices, bias_matrices, x_test are defined
threshold = 0.5
final_predictions = predict_classes(x_test, weight_mat, bias_mat, threshold)

# Convert predictions to one-hot encoded format
encoded_output = final_predictions
df_encoded_output = pd.DataFrame(encoded_output, columns=['setosa', 'versicolor', 'virginica'])

# Decode predictions to class labels
decoded_output = df_encoded_output.idxmax(axis=1)

accuracy = accuracy_score(y_test, decoded_output)
print("Accuracy:", accuracy*100)
