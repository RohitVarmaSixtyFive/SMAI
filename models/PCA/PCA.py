import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.reconstruction_error = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        cov = np.cov(X.T)

        self.eigenvalues, self.eigenvectors = np.linalg.eig(cov)

        idxs = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[idxs]
        self.eigenvectors = self.eigenvectors[:, idxs]  

        self.components = self.eigenvectors[:, :self.n_components]

        self.total_variance = np.sum(self.eigenvalues)

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components)
    
    def inverse_transform(self, X_transformed):
        return np.dot(X_transformed, self.components.T) + self.mean

    def ret_importance(self):
        return self.eigenvalues / np.sum(self.eigenvalues)
    
    def checkPCA(self, X, mse_threshold=0.03, orthogonality_threshold=1e-6):
        self.fit(X)

        dot_products = np.abs(np.dot(self.components.T, self.components) - np.eye(self.n_components))
        is_orthogonal = np.all(dot_products < orthogonality_threshold)

        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        mse = np.mean((X - X_reconstructed) ** 2)
        self.reconstruction_error = mse
        is_reconstruction_accurate = mse <= mse_threshold

        return is_orthogonal and is_reconstruction_accurate