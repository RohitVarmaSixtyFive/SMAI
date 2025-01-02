import numpy as np

class KDE:
    def __init__(self, kernel_type='gaussian', bandwidth=1.0):
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth
        self.data = None
        
    def fit(self, data):
        self.data = np.array(data)
        return self
        
    def kernel(self, u):
        if self.kernel_type == 'box':

            return 0.5 * np.all(np.abs(u) <= 1, axis=-1).astype(float)
        elif self.kernel_type == 'gaussian':

            return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.sum(u ** 2, axis=-1))
        elif self.kernel_type == 'triangular':

            return np.maximum(1 - np.sqrt(np.sum(u ** 2, axis=-1)), 0)
    
    def predict(self, X):
        X = np.array(X)
        

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        n_samples = len(self.data)
        n_features = self.data.shape[1]
        

        diff = (X[:, np.newaxis, :] - self.data[np.newaxis, :, :]) / self.bandwidth
        
        densities = self.kernel(diff)
        
        density = np.sum(densities, axis=1) / (n_samples * self.bandwidth ** n_features)
        
        return density