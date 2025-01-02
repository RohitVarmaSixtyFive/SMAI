import numpy as np

class Kmeans:
    def __init__(self, k, random_state=None):
        self.k = k
        self.random_state = random_state
        self.history_centroids = []  
        self.history_labels = [] 
        
    def fit(self, X_train):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.X_train = X_train
        
        self.centroids = self.X_train[np.random.choice(self.X_train.shape[0], self.k, replace=False)]
        self.labels = self.assign_labels(self.X_train, self.centroids)
        self.cost = self.get_cost(self.X_train, self.centroids)
        
        self.centroids = self.X_train[np.random.choice(self.X_train.shape[0], self.k, replace=False)]
        self.history_centroids.append(self.centroids.copy()) 
        

    def assign_labels(self, X, centroids, max_iter=1000): # the predict function
        prev_distance = float('inf')
        for _ in range(max_iter): 
            
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2) # for doing k distances in one line so its faster but not really required for small k as it doesnt take much time.
            new_labels = np.argmin(distances, axis=1)
            new_centroids = self.update_centroids(X, new_labels)
            
            self.history_centroids.append(new_centroids)
            self.history_labels.append(new_labels)
            
            new_distance = np.sum(np.min(np.linalg.norm(X[:, np.newaxis] - new_centroids, axis=2), axis=1))
            
            if prev_distance - new_distance < 0.0001:
                self.centroids = new_centroids
                return new_labels
            
            prev_distance = new_distance
            centroids = new_centroids

    def update_centroids(self, X, labels):
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
        return new_centroids

    def get_cost(self, X_train, centroids):
        distances = np.linalg.norm(X_train[:, np.newaxis] - centroids, axis=2)
        return np.sum(np.min(distances**2, axis=1))
