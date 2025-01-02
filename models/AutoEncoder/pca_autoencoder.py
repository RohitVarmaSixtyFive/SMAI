import numpy as np

class PcaAutoencoder:
    def __init__(self, n_components):
        """
        Initialize PCA Autoencoder with specified number of components
        
        Args:
            n_components (int): Number of principal components to use
        """
        self.n_components = n_components
        self.eigenvectors = None
        self.mean = None
        self.eigenvalues = None
        
    def fit(self, X):
        """
        Calculate eigenvalues and eigenvectors from the input data
        
        Args:
            X (np.ndarray): Input data matrix of shape (n_samples, n_features)
        """
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        cov_matrix = np.cov(X_centered.T)
        
        self.eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        idx = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        
        self.eigenvectors = self.eigenvectors[:, :self.n_components]
        
    def encode(self, X):
        """
        Reduce dimensionality of input data using learned eigenvectors
        
        Args:
            X (np.ndarray): Input data matrix of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Encoded data with reduced dimensions
        """
        X_centered = X - self.mean
        
        return np.dot(X_centered, self.eigenvectors)
    
    def forward(self, X):
        """
        Reconstruct the input data from its encoded representation
        
        Args:
            X (np.ndarray): Input data matrix of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Reconstructed data in original dimension
        """
        encoded = self.encode(X)
        
        reconstructed = np.dot(encoded, self.eigenvectors.T) + self.mean
        
        return reconstructed
    
    def reconstruction_error(self, X):
        """
        Calculate reconstruction error (MSE) for the input data
        
        Args:
            X (np.ndarray): Input data matrix
            
        Returns:
            float: Mean squared reconstruction error
        """
        reconstructed = self.forward(X)
        return np.mean((X - reconstructed) ** 2)
    
    def explained_variance_ratio(self):
        """
        Calculate the proportion of variance explained by each principal component
        
        Returns:
            np.ndarray: Array of explained variance ratios
        """
        return self.eigenvalues[:self.n_components] / np.sum(self.eigenvalues)