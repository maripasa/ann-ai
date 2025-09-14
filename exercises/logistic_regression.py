from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(n_samples=1000, n_features=2, n_classes=2, n_informative=2, n_redundant=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

bias_train = np.ones((X_train.shape[0], 1))
bias_test = np.ones((X_test.shape[0], 1))
X_train = np.hstack((bias_train, X_train))
X_test = np.hstack((bias_test, X_test))
y_train = 2 * y_train - 1
y_test = 2 * y_test - 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    

#def train():
    #w = np.zeros((3,1))
    #y_train_col = y_train[:, np.newaxis] 
    #n_stocastic = np.array([])
    #for _ in range(100000):
        #aux = X_train @ w
        #aux = sigmoid(y_train_col * aux)
        #aux = y_train_col * aux
        #grad = np.sum(aux*X_train, axis=0, keepdims=True).T
        #w = w - 0.0001 * grad
    #
    #return w

def train():
    w = np.zeros((3,1))
    y_train_col = y_train[:, np.newaxis] 
    for _ in range(100000):
        # Calculate the linear combination
        linear_output = X_train @ w
        
        # Calculate the term for the gradient, which is -y * sigmoid(-y*linear_output)
        grad_term = -y_train_col * sigmoid(-y_train_col * linear_output)
        
        # Calculate the sum of the gradients for each weight
        grad = np.sum(grad_term * X_train, axis=0, keepdims=True).T
        
        # Update the weights
        w = w - 0.0001 * grad
    
    return w

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

w = train()
y_result = sigmoid(X_test @ w)

y_pred = np.where(y_result > 0, 1, -1) 
print(np.mean(y_pred == y_test))


