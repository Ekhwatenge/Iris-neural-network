import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the neural network architecture
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        one_hot_y = np.zeros((m, 3))
        one_hot_y[np.arange(m), y] = 1

        dZ2 = self.a2 - one_hot_y
        dW2 = (1 / m) * np.dot(self.a1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        dZ1 = np.dot(dZ2, self.W2.T) * (self.a1 * (1 - self.a1))
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

# Initialize and train the neural network
nn = NeuralNetwork(input_size=4, hidden_size=10, output_size=3)

epochs = 1000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward pass
    output = nn.forward(X_train)
    
    # Backward pass
    nn.backward(X_train, y_train, learning_rate)
    
    if epoch % 100 == 0:
        loss = -np.mean(np.log(output[np.arange(len(y_train)), y_train]))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Evaluate the model
predictions = np.argmax(nn.forward(X_test), axis=1)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy:.4f}")
