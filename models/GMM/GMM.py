import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, k, max_iter=1000, tolerance=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.means = None
        self.covariances = None
        self.weights = None
        self.count = 0
    
    def init_parameters(self, X):
        n_samples, n_features = X.shape
        
        self.means = self.histeq_select(X, self.k) 
        
        self.covariances = np.array([np.cov(X.T) + 1e-6 * np.eye(n_features) for _ in range(self.k)])
        
        self.weights = np.ones(self.k) / self.k

    
    def histeq_select(self, X, k):
        n_samples, _ = X.shape
        centers = [X[np.random.randint(n_samples)]]
        
        for _ in range(1, k):
            dist = np.array([min([np.linalg.norm(c - x) ** 2 for c in centers]) for x in X])
            probs = dist / dist.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()
            
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    centers.append(X[j])
                    break
        
        return np.array(centers)
    
    def getMembership(self, X):
        n_samples, n_features = X.shape
        responsibilities = np.zeros((n_samples, self.k))
        
        for k in range(self.k):
            try:
                responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(
                    X, mean=self.means[k], cov=self.covariances[k]
                )
            except np.linalg.LinAlgError:
                print(f"U messed up go again")
                responsibilities[:, k] = 0
        
        responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
        responsibilities_sum = np.where(responsibilities_sum == 0, 1, responsibilities_sum)  # Avoid division by zero
        responsibilities /= responsibilities_sum
        
        return responsibilities
    
    def m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        
        self.weights = responsibilities.sum(axis=0) / n_samples
        
        self.means = np.dot(responsibilities.T, X) / responsibilities.sum(axis=0, keepdims=True).T
        
        reg_cov = 1e-6 * np.eye(n_features)  
        for k in range(self.k):
            diff = X - self.means[k]
            self.covariances[k] = np.dot((responsibilities[:, k] * diff.T), diff) / responsibilities[:, k].sum() + reg_cov
    
    def get_params(self):
        return self.means, self.covariances, self.weights
    
    def fit(self, X):
        self.init_parameters(X)
        
        log_likelihood_old = -np.inf
        
        for iteration in range(self.max_iter):

            self.count = iteration
            responsibilities = self.getMembership(X)
            

            self.m_step(X, responsibilities)
            

            log_likelihood_new = self.getLikelihood(X)
            

            if np.abs(log_likelihood_new - log_likelihood_old) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            log_likelihood_old = log_likelihood_new
    
    def getLikelihood(self, X):
        n_samples = X.shape[0]
        likelihood = np.zeros((n_samples, self.k))
        
        for k in range(self.k):
            try:
                likelihood[:, k] = self.weights[k] * multivariate_normal.pdf(
                    X, mean=self.means[k], cov=self.covariances[k]
                )
            except np.linalg.LinAlgError:

                print(f"Covariance matrix for component {k} is not positive definite")
                likelihood[:, k] = 1e-10  
        

        return np.sum(np.log(np.sum(likelihood + 1e-10, axis=1)))
    
    def predict(self, X):
        responsibilities = self.getMembership(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X):
        return self.getMembership(X)
