import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))  
from models.MLP.MLP import MLP_Reg
from sklearn.metrics import mean_squared_error, r2_score

class AutoEncoder:
    def __init__(self, input_dim, architecture=None, learning_rate=0.001, 
                 epochs=1000, batch_size=32):
        """
        Initialize AutoEncoder with basic parameters.
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.architecture = architecture if architecture is not None else [
            {"units": 12, "activation": "relu"},    
            {"units": 9, "activation": "relu"},   
            {"units": 12, "activation": "relu"},
        ]
        
        self.architecture.append({"units": input_dim, "activation": "linear"})
        self.model = None  

    def _build_model(self, X):
        """
        Build the MLP model based on the defined architecture.
        """
        hidden_layers = [layer["units"] for layer in self.architecture[:-1]] 
        return MLP_Reg(
            hidden_neurons=hidden_layers, 
            num_hid_layers=len(hidden_layers), 
            epochs=self.epochs, 
            learning_rate=self.learning_rate, 
            activation='relu', 
            optimizer='mini_batch_gd', 
            batch_size=self.batch_size,
            output_size=X.shape[1]  
        )

    def fit(self, X):
        """
        Fit the AutoEncoder model on the input data X.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        self.model = self._build_model(X)
        
        self.model.train(X, X)
        
        return self.model.train_loss  

    def get_latent(self, X):
        """
        Extract the latent representation (middle layer) from the model.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        latent_layer_index = len(self.architecture) // 2

        activations, _ = self.model.forward_propagation(X)
        
        return activations[latent_layer_index]

    def reconstruct(self, X):
        """
        Reconstruct the input X using the trained model.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
