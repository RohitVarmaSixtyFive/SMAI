import numpy as np
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import copy  
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from typing import List, Tuple
import numpy as np


class MLP_SingleLabelClassifier:
    
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, activation='relu', optimizer='sgd', batch_size=32, epochs=100):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.initialize_weights()

    def initialize_weights(self):
        for i in range(1, len(self.layers)):
            self.weights.append(np.random.randn(self.layers[i-1], self.layers[i]) * np.sqrt(2. / self.layers[i-1]))
            self.biases.append(np.zeros((1, self.layers[i])))

    def activate(self, Z, derivative=False):
        """
        Activation function handler to compute both the activation and its derivative.
        """
        if self.activation == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-Z))
            if derivative:
                return sigmoid * (1 - sigmoid)
            return sigmoid
        elif self.activation == 'relu':
            if derivative:
                return np.where(Z > 0, 1, 0)
            return np.maximum(0, Z)
        elif self.activation == 'tanh':
            if derivative:
                return 1 - np.tanh(Z) ** 2  
            return np.tanh(Z)
        elif self.activation == 'linear':
            if derivative:
                return np.ones_like(Z)  
            return Z
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

    def softmax(self, Z): # for the last layer as it requires probabilities
        exp_scores = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def forward_propagation(self, X):
        self.Z = []
        self.A = [X]
        
        for i in range(len(self.weights)):
            Z = np.dot(self.A[-1], self.weights[i]) + self.biases[i]
            self.Z.append(Z)
            
            if i == len(self.weights) - 1: 
                A = self.softmax(Z)  
            else:
                A = self.activate(Z)
            self.A.append(A)
            
        return self.A[-1]

    def backward_propagation(self, X, y):
        m = X.shape[0]
        y_encoded = np.eye(self.output_size)[y]
        
        dZ_out = self.A[-1] - y_encoded
        
        dW = [1/m * np.dot(self.A[-2].T, dZ_out)]
        db = [1/m * np.sum(dZ_out, axis=0, keepdims=True)]
        
        dZ = np.dot(dZ_out, self.weights[-1].T) * self.activate(self.Z[-2], derivative=True)
        
        for i in range(len(self.weights) - 2, -1, -1):
            dW_i = 1/m * np.dot(self.A[i].T, dZ)
            db_i = 1/m * np.sum(dZ, axis=0, keepdims=True)
            
            dW.insert(0, dW_i)
            db.insert(0, db_i)
            
            if i > 0:
                dZ = np.dot(dZ, self.weights[i].T) * self.activate(self.Z[i-1], derivative=True)

        return dW, db

    def update_parameters(self, dW, db): 
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        log_probs = -np.log(y_pred[range(m), y_true])
        loss = np.sum(log_probs) / m
        return loss

    def fit(self, X, y, wandb=False, validation=False, X_val=None, y_val=None, early_stopping_patience=10, min_delta=1e-3):
        
        X, y = np.array(X), np.array(y)
        if validation:
            X_val, y_val = np.array(X_val), np.array(y_val)
        
        best_val_loss = float('inf')
        patience_counter = 0  
        
        for epoch in range(self.epochs):
            if self.optimizer in ['batch_gd', 'mini_batch_gd']:
                indices = np.arange(X.shape[0])
                np.random.shuffle(indices)
                X, y = X[indices], y[indices]
            
            if self.optimizer == 'batch_gd':
                y_pred = self.forward_propagation(X)
                dW, db = self.backward_propagation(X, y)
                self.update_parameters(dW, db)
                
            elif self.optimizer == 'sgd':
                for i in range(X.shape[0]):
                    X_batch, y_batch = X[i:i+1], y[i:i+1]
                    y_pred = self.forward_propagation(X_batch)
                    dW, db = self.backward_propagation(X_batch, y_batch)
                    self.update_parameters(dW, db)
                    
            elif self.optimizer == 'mini_batch_gd':
                for i in range(0, X.shape[0], self.batch_size):
                    X_batch = X[i:i+self.batch_size]
                    y_batch = y[i:i+self.batch_size]
                    y_pred = self.forward_propagation(X_batch)
                    
                    dW, db = self.backward_propagation(X_batch, y_batch)
                    self.update_parameters(dW, db)
            
            train_pred = self.predict(X)
            train_loss = self.compute_loss(y, self.predict_proba(X))
            train_acc = accuracy_score(y, train_pred)

            if validation:
                val_pred = self.predict(X_val)
                val_loss = self.compute_loss(y_val, self.predict_proba(X_val))
                val_acc = accuracy_score(y_val, val_pred)

                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                if wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'f1_score': f1_score(y_val, val_pred, average='weighted'),
                        'precision': precision_score(y_val, val_pred, average='weighted'),
                        'recall': recall_score(y_val, val_pred, average='weighted')
                    })

                if val_loss < best_val_loss - min_delta:  
                    best_val_loss = val_loss
                    patience_counter = 0  
                else:
                    patience_counter += 1  

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f}")
                    break
            else:
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)

    def predict(self, X):
        probabilities = self.forward_propagation(X)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, X):
        return self.forward_propagation(X)

    def gradient_check(self, X, y, epsilon=1e-7):
        X = np.array(X)
        y = np.array(y)
        y_encoded = np.eye(self.output_size)[y]

        def compute_cost(X, y_encoded):
            y_pred = self.forward_propagation(X)
            return -np.sum(y_encoded * np.log(y_pred + 1e-8)) / y_encoded.shape[0]

        y_pred = self.forward_propagation(X)
        dW, db = self.backward_propagation(X, y)
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            dW_numerical = np.zeros_like(w)
            db_numerical = np.zeros_like(b)
            
            it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                index = it.multi_index
                old_value = w[index]
                w[index] = old_value + epsilon
                cost_plus = compute_cost(X, y_encoded)
                w[index] = old_value - epsilon
                cost_minus = compute_cost(X, y_encoded)
                w[index] = old_value
                dW_numerical[index] = (cost_plus - cost_minus) / (2 * epsilon)
                it.iternext()
            
            it = np.nditer(b, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                index = it.multi_index
                old_value = b[index]
                b[index] = old_value + epsilon
                cost_plus = compute_cost(X, y_encoded)
                b[index] = old_value - epsilon
                cost_minus = compute_cost(X, y_encoded)
                b[index] = old_value
                db_numerical[index] = (cost_plus - cost_minus) / (2 * epsilon)
                it.iternext()
            
            print(f"Layer {i+1}:")
            print(f"  Weights - Max difference: {np.max(np.abs(dW[i] - dW_numerical))}")
            print(f"  Biases - Max difference: {np.max(np.abs(db[i] - db_numerical))}")
            print(f"  Relative difference (weights): {np.linalg.norm(dW[i] - dW_numerical) / (np.linalg.norm(dW[i]) + np.linalg.norm(dW_numerical))}")
            print(f"  Relative difference (biases): {np.linalg.norm(db[i] - db_numerical) / (np.linalg.norm(db[i]) + np.linalg.norm(db_numerical))}")

  
class MLP_MultiLabelClassifier:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, activation='relu', optimizer='mini_batch_gd', batch_size=32, epochs=100):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.weights = []
        self.biases = []
        
        # Initialize layer sizes
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i]) if activation == 'relu' else np.sqrt(1.0 / layer_sizes[i])
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
        
        # Adam optimizer parameters
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.timestep = 0
        
        # Store training/validation loss
        self.training_loss = []
        self.validation_loss = []

    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -15, 15)))
    
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward_pass(self, X):
        activations = [X]
        z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            
            # Batch normalization for hidden layers
            if i < len(self.weights) - 1:
                z = (z - np.mean(z, axis=0, keepdims=True)) / (np.std(z, axis=0, keepdims=True) + 1e-8)
            
            z_values.append(z)
            
            # Apply activation: ReLU for hidden layers, Sigmoid for output layer
            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = self.sigmoid(z)
            activations.append(a)
        
        return activations, z_values
    
    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward_pass(self, X, y, activations, z_values):
        m = X.shape[0]
        delta = activations[-1] - y
        
        weight_grads = []
        bias_grads = []
        
        for i in range(len(self.weights) - 1, -1, -1):
            weight_grad = np.dot(activations[i].T, delta) / m
            bias_grad = np.sum(delta, axis=0, keepdims=True) / m
            
            # Gradient clipping for stability
            weight_grad = np.clip(weight_grad, -1, 1)
            bias_grad = np.clip(bias_grad, -1, 1)
            
            weight_grads.insert(0, weight_grad)
            bias_grads.insert(0, bias_grad)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                if self.activation == 'relu':
                    delta *= self.relu_derivative(z_values[i - 1])
                delta = np.clip(delta, -1, 1)  # Gradient clipping
        
        return weight_grads, bias_grads
    
    def update_parameters_adam(self, weight_grads, bias_grads):
        self.timestep += 1
        
        for i in range(len(self.weights)):
            # Update weights using Adam optimizer
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * weight_grads[i]
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * np.square(weight_grads[i])
            
            m_hat = self.m_weights[i] / (1 - self.beta1 ** self.timestep)
            v_hat = self.v_weights[i] / (1 - self.beta2 ** self.timestep)
            
            self.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            # Update biases using Adam optimizer
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * bias_grads[i]
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * np.square(bias_grads[i])
            
            m_hat_bias = self.m_biases[i] / (1 - self.beta1 ** self.timestep)
            v_hat_bias = self.v_biases[i] / (1 - self.beta2 ** self.timestep)
            
            self.biases[i] -= self.learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)
    
    def fit(self, X, y, validation=False, X_val=None, y_val=None):
        # Normalize input data
        X = (X - np.mean(X, axis=0, keepdims=True)) / (np.std(X, axis=0, keepdims=True) + 1e-8)
        if validation:
            X_val = (X_val - np.mean(X_val, axis=0, keepdims=True)) / (np.std(X_val, axis=0, keepdims=True) + 1e-8)
        
        n_samples = X.shape[0]
        best_loss = float('inf')
        patience = 20  # Early stopping patience
        patience_counter = 0
        min_lr = 1e-6
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            
            for start_idx in range(0, n_samples, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                activations, z_values = self.forward_pass(X_batch)
                weight_grads, bias_grads = self.backward_pass(X_batch, y_batch, activations, z_values)
                self.update_parameters_adam(weight_grads, bias_grads)
            
            # Calculate training loss
            activations, _ = self.forward_pass(X)
            train_loss = self.compute_loss(y, activations[-1])
            self.training_loss.append(train_loss)
            
            # Early stopping
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Learning rate decay
            if patience_counter > patience // 2:
                self.learning_rate = max(self.learning_rate * 0.9, min_lr)
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Validation
            if validation and X_val is not None and y_val is not None:
                val_activations, _ = self.forward_pass(X_val)
                val_loss = self.compute_loss(y_val, val_activations[-1])
                self.validation_loss.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {train_loss:.4f}", end="")
                if validation:
                    print(f", Validation Loss: {val_loss:.4f}")
                else:
                    print()
    
    def predict(self, X):
        X = (X - np.mean(X, axis=0, keepdims=True)) / (np.std(X, axis=0, keepdims=True) + 1e-8)
        activations, _ = self.forward_pass(X)
        return (activations[-1] >= 0.5).astype(int)


class MLP_Regressor:
    
    def __init__(self, input_size, hidden_layers, output_size, 
                 learning_rate=0.01, activation='relu', optimizer='sgd', 
                 batch_size=32, epochs=100, patience=10):
        """
        Initializes MLP Regressor with specified parameters.
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        # Define network architecture
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []  # Store weights for each layer
        self.biases = []   # Store biases for each layer
        self.history = {'train_loss': [], 'val_loss': []}  # Track losses over time
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize weights using He initialization for ReLU activation.
        """
        for i in range(1, len(self.layers)):
            if self.activation == 'relu':
                self.weights.append(np.random.randn(self.layers[i-1], self.layers[i]) * np.sqrt(2. / self.layers[i-1]))
            else:
                self.weights.append(np.random.randn(self.layers[i-1], self.layers[i]) * np.sqrt(1. / self.layers[i-1]))
            self.biases.append(np.zeros((1, self.layers[i])))  # Initialize biases as zeros

    def activate(self, Z, derivative=False):
        """
        Applies the activation function and its derivative.
        """
        if self.activation == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-Z))
            return sigmoid * (1 - sigmoid) if derivative else sigmoid
        elif self.activation == 'relu':
            return np.where(Z > 0, 1, 0) if derivative else np.maximum(0, Z)
        elif self.activation == 'tanh':
            return 1 - np.tanh(Z) ** 2 if derivative else np.tanh(Z)
        elif self.activation == 'linear':  # For the output layer in regression
            return np.ones_like(Z) if derivative else Z

    def forward_propagation(self, X):
        """
        Perform forward propagation through the network.
        """
        self.Z = []  
        self.A = [X] 
        
        for i in range(len(self.weights)):
            Z = np.dot(self.A[-1], self.weights[i]) + self.biases[i]  # Z = W * A + b
            self.Z.append(Z)  
            
            if i == len(self.weights) - 1:
                A = Z  
            else:
                A = self.activate(Z) 
            self.A.append(A)
            
        return self.A[-1] 

    def backward_propagation(self, X, y):
        """
        Perform backward propagation to compute gradients.
        """
        m = X.shape[0]  
        
        dZ_out = self.A[-1] - y  
        
        dW = [1/m * np.dot(self.A[-2].T, dZ_out)]
        db = [1/m * np.sum(dZ_out, axis=0, keepdims=True)]

        dZ = dZ_out
        for i in range(len(self.weights) - 2, -1, -1):
            dZ = np.dot(dZ, self.weights[i+1].T) * self.activate(self.Z[i], derivative=True)
            dW_i = 1/m * np.dot(self.A[i].T, dZ)
            db_i = 1/m * np.sum(dZ, axis=0, keepdims=True)
            dW.insert(0, dW_i)
            db.insert(0, db_i)

        return dW, db
    
    def update_parameters(self, dW, db):
        """
        Updates weights and biases using gradient descent.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def compute_loss(self, y_true, y_pred):
        """
        Computes Mean Squared Error (MSE) loss.
        """
        return np.mean((y_true - y_pred) ** 2)  # MSE

    def fit(self, X, y, validation_split=0.2):
        """
        Train the MLP model using the specified optimizer and batch size.
        """
        X, y = np.array(X), np.array(y)
        
        if validation_split > 0:
            split_idx = int(X.shape[0] * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            if self.optimizer in ['batch_gd', 'mini_batch_gd']:
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                X_train, y_train = X_train[indices], y_train[indices]
            
            if self.optimizer == 'batch_gd':
                y_pred = self.forward_propagation(X_train)
                dW, db = self.backward_propagation(X_train, y_train)
                self.update_parameters(dW, db)
                
            elif self.optimizer == 'sgd':
                for i in range(X_train.shape[0]):
                    X_batch, y_batch = X_train[i:i+1], y_train[i:i+1]
                    y_pred = self.forward_propagation(X_batch)
                    dW, db = self.backward_propagation(X_batch, y_batch)
                    self.update_parameters(dW, db)
                    
            elif self.optimizer == 'mini_batch_gd':
                for i in range(0, X_train.shape[0], self.batch_size):
                    X_batch = X_train[i:i+self.batch_size]
                    y_batch = y_train[i:i+self.batch_size]
                    y_pred = self.forward_propagation(X_batch)
                    dW, db = self.backward_propagation(X_batch, y_batch)
                    self.update_parameters(dW, db)
            
            train_loss = self.compute_loss(y_train, self.predict(X_train))
            self.history['train_loss'].append(train_loss)
            
            if validation_split > 0:
                val_loss = self.compute_loss(y_val, self.predict(X_val))
                self.history['val_loss'].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f}")
                    break
                
            if (epoch + 1) % 100 == 1:
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}" + 
                      (f", Val Loss: {val_loss:.4f}" if validation_split > 0 else ""))

    def predict(self, X):
        """
        Predict output for the given input.
        """
        return self.forward_propagation(X)
    
    def check_gradients(self, X, y, epsilon=1e-7):
        """
        Performs gradient checking to verify correctness of backpropagation.
        """
        X = np.array(X)
        y = np.array(y)
        
        y_pred = self.forward_propagation(X)
        dW, db = self.backward_propagation(X, y)

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            dW_numerical = np.zeros_like(w)
            db_numerical = np.zeros_like(b)

            it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                index = it.multi_index
                old_value = w[index]

                w[index] = old_value + epsilon
                cost_plus = self.compute_loss(y, self.forward_propagation(X))

                w[index] = old_value - epsilon
                cost_minus = self.compute_loss(y, self.forward_propagation(X))

                w[index] = old_value

                dW_numerical[index] = (cost_plus - cost_minus) / (2 * epsilon)
                it.iternext()

            it = np.nditer(b, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                index = it.multi_index
                old_value = b[index]

                b[index] = old_value + epsilon
                cost_plus = self.compute_loss(y, self.forward_propagation(X))

                b[index] = old_value - epsilon
                cost_minus = self.compute_loss(y, self.forward_propagation(X))

                b[index] = old_value

                db_numerical[index] = (cost_plus - cost_minus) / (2 * epsilon)
                it.iternext()

            print(f"Layer {i+1}:")
            print(f"  Weights - Max difference: {np.max(np.abs(dW[i] - dW_numerical))}")
            print(f"  Biases - Max difference: {np.max(np.abs(db[i] - db_numerical))}")
            print(f"  Relative difference (weights): {np.linalg.norm(dW[i] - dW_numerical) / (np.linalg.norm(dW[i]) + np.linalg.norm(dW_numerical))}")
            print(f"  Relative difference (biases): {np.linalg.norm(db[i] - db_numerical) / (np.linalg.norm(db[i]) + np.linalg.norm(db_numerical))}")


class MLP_Reg:

    def __init__(self, hidden_neurons, num_hid_layers, epochs, output_size=1,
                 learning_rate=0.01, activation='tanh', optimizer='batch_gd', 
                 batch_size=None):
        self.hidden_layer_sizes = hidden_neurons
        self.num_hidden = num_hid_layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.output_size = output_size
        self.train_loss = []
        self.batch_size = batch_size
        if optimizer == "sgd":
            self.batch_size = 1
        elif optimizer == "batch_gd":
            self.batch_size = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid_derivative(self, activations):
        return activations * (1 - activations)

    def tanh_derivative(self, activations):
        return 1 - np.power(activations, 2)

    def relu_derivative(self, activations):
        return np.where(activations > 0, 1, 0)

    def activation_function(self, weighted_sums):
        if self.activation == "sigmoid":
            return self.sigmoid(weighted_sums)
        elif self.activation == "tanh":
            return self.tanh(weighted_sums)
        elif self.activation == "relu":
            return self.relu(weighted_sums)
    
    def activation_derivative(self, activations):
        if self.activation == "sigmoid":
            return self.sigmoid_derivative(activations)
        elif self.activation == "tanh":
            return self.tanh_derivative(activations)
        elif self.activation == "relu":
            return self.relu_derivative(activations)

    def forward_propagation(self, X):
        activations = [X]
        weighted_sums = []
        for i in range(len(self.weights) - 1):
            arr = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            weighted_sums.append(arr)
            activations.append(self.activation_function(arr))
        arr = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        activations.append(arr)
        weighted_sums.append(arr)
        return activations, weighted_sums

    def MSE(self, Y_true, Y_pred):
        loss = np.mean((Y_true - Y_pred) ** 2)
        return loss

    def backward_propagation(self, X, y, activations, weighted_sums):
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]
        y = y.reshape((-1, self.output_size))
        delta = 2 * (activations[-1] - y)
        gradients_w[-1] = np.dot(activations[-2].T, delta)
        gradients_b[-1] = np.sum(delta, axis=0, keepdims=True)

        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(delta, self.weights[i + 1].T) * self.activation_derivative(activations[i + 1])
            gradients_w[i] = np.dot(activations[i].T, delta)
            gradients_b[i] = np.sum(delta, axis=0, keepdims=True)

        for i in range(len(self.weights)):
            self.weights[i] -= (self.learning_rate * gradients_w[i]) / self.batch_size
            self.biases[i] -= (self.learning_rate * gradients_b[i]) / self.batch_size

    def train(self, X, Y):
        X = X.to_numpy()
        self.input_size = X.shape[1]
        layer_sizes = np.hstack((np.array([self.input_size]), self.hidden_layer_sizes, np.array([self.output_size])))
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) / np.sqrt(layer_sizes[i + 1]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

        if self.batch_size is None or self.batch_size > X.shape[0]:
            self.batch_size = X.shape[0]

        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        Y = Y.to_numpy()
        X = X[indices]
        Y = Y[indices]

        for j in range(self.epochs):
            loss_arr = []

            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i + self.batch_size]
                Y_batch = Y[i:i + self.batch_size]

                activations, weighted_sums = self.forward_propagation(X_batch)
                self.backward_propagation(X_batch, Y_batch, activations, weighted_sums)
                loss_arr.append(self.MSE(Y_batch, activations[-1]))
            loss = np.mean(loss_arr)
            self.train_loss.append(loss)
        

    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        return activations[-1]
 
 
class MLP_BCE_Regressor:
    
    def __init__(self, input_size, hidden_layers, output_size, 
                 learning_rate=0.01, activation='relu', optimizer='sgd', 
                 batch_size=32, epochs=100, patience=10, loss='mse'):
        """
        Initializes MLP Regressor with specified parameters.
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.loss = loss
        
        # Define network architecture
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []  # Store weights for each layer
        self.biases = []   # Store biases for each layer
        self.history = {'train_loss': [], 'val_loss': []}  # Track losses over time
        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize weights using He initialization for ReLU activation.
        """
        for i in range(1, len(self.layers)):
            if self.activation == 'relu':
                self.weights.append(np.random.randn(self.layers[i-1], self.layers[i]) * np.sqrt(2. / self.layers[i-1]))
            else:
                self.weights.append(np.random.randn(self.layers[i-1], self.layers[i]) * np.sqrt(1. / self.layers[i-1]))
            self.biases.append(np.zeros((1, self.layers[i])))  # Initialize biases as zeros

    def activate(self, Z, derivative=False):
        """
        Applies the activation function and its derivative.
        """
        if self.activation == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-Z))
            return sigmoid * (1 - sigmoid) if derivative else sigmoid
        elif self.activation == 'relu':
            return np.where(Z > 0, 1, 0) if derivative else np.maximum(0, Z)
        elif self.activation == 'tanh':
            return 1 - np.tanh(Z) ** 2 if derivative else np.tanh(Z)
        elif self.activation == 'linear':  # For the output layer in regression
            return np.ones_like(Z) if derivative else Z

    def forward_propagation(self, X):
        """
        Perform forward propagation through the network.
        """
        self.Z = []  
        self.A = [X] 
        
        for i in range(len(self.weights)):
            Z = np.dot(self.A[-1], self.weights[i]) + self.biases[i]  # Z = W * A + b
            self.Z.append(Z)  
            
            if i == len(self.weights) - 1:
                A = self.sigmoid(Z) if self.loss == 'bce' else Z
            else:
                A = self.activate(Z) 
            self.A.append(A)
            
        return self.A[-1] 

    def backward_propagation(self, X, y):
        """
        Perform backward propagation to compute gradients.
        """
        m = X.shape[0]  
        
        if self.loss == 'mse':
            dZ_out = self.A[-1] - y
        elif self.loss == 'bce':
            dZ_out = self.A[-1] - y
        
        dW = [1/m * np.dot(self.A[-2].T, dZ_out)]
        db = [1/m * np.sum(dZ_out, axis=0, keepdims=True)]

        dZ = dZ_out
        for i in range(len(self.weights) - 2, -1, -1):
            dZ = np.dot(dZ, self.weights[i+1].T) * self.activate(self.Z[i], derivative=True)
            dW_i = 1/m * np.dot(self.A[i].T, dZ)
            db_i = 1/m * np.sum(dZ, axis=0, keepdims=True)
            dW.insert(0, dW_i)
            db.insert(0, db_i)

        return dW, db
    
    def update_parameters(self, dW, db):
        """
        Updates weights and biases using gradient descent.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def compute_loss(self, y_true, y_pred):
        """
        Computes loss based on the specified loss function.
        """
        if self.loss == 'mse':
            return np.mean((y_true - y_pred) ** 2)  # MSE
        elif self.loss == 'bce':
            epsilon = 1e-15  # Small value to avoid log(0)
            return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

    def fit(self, X, y, validation_split=0.2):
        """
        Train the MLP model using the specified optimizer and batch size.
        """
        X, y = np.array(X), np.array(y)
        
        if validation_split > 0:
            split_idx = int(X.shape[0] * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            if self.optimizer in ['batch_gd', 'mini_batch_gd']:
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                X_train, y_train = X_train[indices], y_train[indices]
            
            if self.optimizer == 'batch_gd':
                y_pred = self.forward_propagation(X_train)
                dW, db = self.backward_propagation(X_train, y_train)
                self.update_parameters(dW, db)
                
            elif self.optimizer == 'sgd':
                for i in range(X_train.shape[0]):
                    X_batch, y_batch = X_train[i:i+1], y_train[i:i+1]
                    y_pred = self.forward_propagation(X_batch)
                    dW, db = self.backward_propagation(X_batch, y_batch)
                    self.update_parameters(dW, db)
                    
            elif self.optimizer == 'mini_batch_gd':
                for i in range(0, X_train.shape[0], self.batch_size):
                    X_batch = X_train[i:i+self.batch_size]
                    y_batch = y_train[i:i+self.batch_size]
                    y_pred = self.forward_propagation(X_batch)
                    dW, db = self.backward_propagation(X_batch, y_batch)
                    self.update_parameters(dW, db)
            
            train_loss = self.compute_loss(y_train, self.predict(X_train))
            self.history['train_loss'].append(train_loss)
            
            if validation_split > 0:
                val_loss = self.compute_loss(y_val, self.predict(X_val))
                self.history['val_loss'].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f}")
                    break
                
            if (epoch + 1) % 100 == 1:
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}" + 
                      (f", Val Loss: {val_loss:.4f}" if validation_split > 0 else ""))

    def predict(self, X):
        """
        Predict output for the given input.
        """
        return self.forward_propagation(X)
    
    def check_gradients(self, X, y, epsilon=1e-7):
        """
        Performs gradient checking to verify correctness of backpropagation.
        """
        X = np.array(X)
        y = np.array(y)
        
        y_pred = self.forward_propagation(X)
        dW, db = self.backward_propagation(X, y)

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            dW_numerical = np.zeros_like(w)
            db_numerical = np.zeros_like(b)

            it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                index = it.multi_index
                old_value = w[index]

                w[index] = old_value + epsilon
                cost_plus = self.compute_loss(y, self.forward_propagation(X))

                w[index] = old_value - epsilon
                cost_minus = self.compute_loss(y, self.forward_propagation(X))

                w[index] = old_value

                dW_numerical[index] = (cost_plus - cost_minus) / (2 * epsilon)
                it.iternext()

            it = np.nditer(b, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                index = it.multi_index
                old_value = b[index]

                b[index] = old_value + epsilon
                cost_plus = self.compute_loss(y, self.forward_propagation(X))

                b[index] = old_value - epsilon
                cost_minus = self.compute_loss(y, self.forward_propagation(X))

                b[index] = old_value

                db_numerical[index] = (cost_plus - cost_minus) / (2 * epsilon)
                it.iternext()

            print(f"Layer {i+1}:")
            print(f"  Weights - Max difference: {np.max(np.abs(dW[i] - dW_numerical))}")
            print(f"  Biases - Max difference: {np.max(np.abs(db[i] - db_numerical))}")
            print(f"  Relative difference (weights): {np.linalg.norm(dW[i] - dW_numerical) / (np.linalg.norm(dW[i]) + np.linalg.norm(dW_numerical))}")
            print(f"  Relative difference (biases): {np.linalg.norm(db[i] - db_numerical) / (np.linalg.norm(db[i]) + np.linalg.norm(db_numerical))}")

    def sigmoid(self, Z):
        """
        Compute the sigmoid activation function.
        """
        return 1 / (1 + np.exp(-Z)) 
 
   
class MLP:
    def __init__(self, input_size, hidden_layers, output_size, task_type='classifier',
                 learning_rate=0.01, activation='relu', optimizer='sgd', 
                 batch_size=32, epochs=100, patience=10):

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        
        if self.task_type == 'classifier':
            self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        else:
            self.history = {'train_loss': [], 'val_loss': []}
            
        self.initialize_weights()

    def initialize_weights(self):
        for i in range(1, len(self.layers)):
            self.weights.append(np.random.randn(self.layers[i-1], self.layers[i]) * np.sqrt(2. / self.layers[i-1]))
            self.biases.append(np.zeros((1, self.layers[i])))

    def activate(self, Z, derivative=False):
        if self.activation == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-Z))
            if derivative:
                return sigmoid * (1 - sigmoid)
            return sigmoid
        elif self.activation == 'relu':
            if derivative:
                return np.where(Z > 0, 1, 0)
            return np.maximum(0, Z)
        elif self.activation == 'tanh':
            if derivative:
                return 1 - np.tanh(Z) ** 2  
            return np.tanh(Z)
        elif self.activation == 'linear':
            if derivative:
                return np.ones_like(Z)  
            return Z
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

    def softmax(self, Z):
        exp_scores = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def forward_propagation(self, X):
        self.Z = []
        self.A = [X]
        
        for i in range(len(self.weights)):
            Z = np.dot(self.A[-1], self.weights[i]) + self.biases[i]
            self.Z.append(Z)
            
            if i == len(self.weights) - 1:
                if self.task_type == 'classifier':
                    A = self.softmax(Z)
                else:
                    A = Z  
            else:
                A = self.activate(Z)
            self.A.append(A)
            
        return self.A[-1]

    def backward_propagation(self, X, y):
        m = X.shape[0]
        
        if self.task_type == 'classifier':
            y_encoded = np.eye(self.output_size)[y]
            dZ_out = self.A[-1] - y_encoded
        else:
            dZ_out = self.A[-1] - y  # For MSE loss
        
        dW = [1/m * np.dot(self.A[-2].T, dZ_out)]
        db = [1/m * np.sum(dZ_out, axis=0, keepdims=True)]
        
        dZ = np.dot(dZ_out, self.weights[-1].T) * self.activate(self.Z[-2], derivative=True)
        
        for i in range(len(self.weights) - 2, -1, -1):
            dW_i = 1/m * np.dot(self.A[i].T, dZ)
            db_i = 1/m * np.sum(dZ, axis=0, keepdims=True)
            
            dW.insert(0, dW_i)
            db.insert(0, db_i)
            
            if i > 0:
                dZ = np.dot(dZ, self.weights[i].T) * self.activate(self.Z[i-1], derivative=True)

        return dW, db

    def update_parameters(self, dW, db):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def compute_loss(self, y_true, y_pred):
        if self.task_type == 'classifier':
            m = y_true.shape[0]
            y_encoded = np.eye(self.output_size)[y_true]
            return -np.sum(y_encoded * np.log(y_pred + 1e-8)) / m
        else:
            return np.mean((y_true - y_pred) ** 2)  # MSE loss

    def compute_accuracy(self, y_true, y_pred):
        if self.task_type == 'classifier':
            return np.mean(y_true == np.argmax(y_pred, axis=1))
        else:
            return None

    def fit(self, X, y, validation_split=0.2):
        X, y = np.array(X), np.array(y)
        
        # Split data for validation
        if validation_split > 0:
            split_idx = int(X.shape[0] * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            if self.optimizer in ['batch_gd', 'mini_batch_gd']:
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                X_train, y_train = X_train[indices], y_train[indices]
            
            if self.optimizer == 'batch_gd':
                y_pred = self.forward_propagation(X_train)
                dW, db = self.backward_propagation(X_train, y_train)
                self.update_parameters(dW, db)
                
            elif self.optimizer == 'sgd':
                for i in range(X_train.shape[0]):
                    X_batch, y_batch = X_train[i:i+1], y_train[i:i+1]
                    y_pred = self.forward_propagation(X_batch)
                    dW, db = self.backward_propagation(X_batch, y_batch)
                    self.update_parameters(dW, db)
                    
            elif self.optimizer == 'mini_batch_gd':
                for i in range(0, X_train.shape[0], self.batch_size):
                    X_batch = X_train[i:i+self.batch_size]
                    y_batch = y_train[i:i+self.batch_size]
                    y_pred = self.forward_propagation(X_batch)
                    dW, db = self.backward_propagation(X_batch, y_batch)
                    self.update_parameters(dW, db)
            
            # Compute training metrics
            train_pred = self.predict(X_train)
            train_loss = self.compute_loss(y_train, train_pred)
            self.history['train_loss'].append(train_loss)
            
            if self.task_type == 'classifier':
                train_acc = self.compute_accuracy(y_train, train_pred)
                self.history['train_acc'].append(train_acc)
            
            # Compute validation metrics
            if validation_split > 0:
                val_pred = self.predict(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                self.history['val_loss'].append(val_loss)
                
                if self.task_type == 'classifier':
                    val_acc = self.compute_accuracy(y_val, val_pred)
                    self.history['val_acc'].append(val_acc)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f}")
                    break
            
            if (epoch + 1) % 100 == 0:
                if self.task_type == 'classifier':
                    print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}" +
                          (f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}" if validation_split > 0 else ""))
                else:
                    print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}" +
                          (f", Val Loss: {val_loss:.4f}" if validation_split > 0 else ""))

    def predict(self, X):
        predictions = self.forward_propagation(X)
        if self.task_type == 'classifier':
            return predictions
        return predictions

    def predict_classes(self, X):

        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)