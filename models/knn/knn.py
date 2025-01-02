import numpy as np
import numpy as np

class KNNClassifier:
    def __init__(self, k=5, distance_metric='euclidean', batch_size=100):
        self.k = k
        self.distance_metric = distance_metric
        self.batch_size = batch_size  
        

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _calculate_distances_batch(self, X_test_batch):

        if self.distance_metric == 'euclidean':
            X_train_sq = np.sum(self.X_train ** 2, axis=1)
            X_test_sq = np.sum(X_test_batch ** 2, axis=1)
            cross_term = np.dot(X_test_batch, self.X_train.T)
            distances = np.sqrt(X_test_sq[:, np.newaxis] - 2 * cross_term + X_train_sq)
            
        elif self.distance_metric == 'manhattan':
            distances = np.abs(X_test_batch[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]).sum(axis=2)
            
        elif self.distance_metric == 'cosine':
            norm_X_test = np.linalg.norm(X_test_batch, axis=1)
            norm_X_train = np.linalg.norm(self.X_train, axis=1)
            dot_product = np.dot(X_test_batch, self.X_train.T)
            cosine_similarity = dot_product / (norm_X_test[:, np.newaxis] * norm_X_train)
            distances = 1 - cosine_similarity
        return distances

    def _get_nearest_neighbors(self, distances):

        return np.argsort(distances, axis=1)[:, :self.k]

    def _vote(self, k_nearest_labels):

        num_test = k_nearest_labels.shape[0]
        predictions = np.zeros(num_test, dtype=int)
        for i in range(num_test):
            labels = k_nearest_labels[i]
            counts = np.bincount(labels)
            predictions[i] = np.argmax(counts)
        return predictions

    def predict(self, X_test):
        num_test = X_test.shape[0]
        predictions = np.zeros(num_test, dtype=int)
        

        for start in range(0, num_test, self.batch_size):
            end = min(start + self.batch_size, num_test)
            X_test_batch = X_test[start:end]
            
            distances = self._calculate_distances_batch(X_test_batch)
            k_nearest_indices = self._get_nearest_neighbors(distances)
            k_nearest_labels = self.y_train[k_nearest_indices]
            predictions[start:end] = self._vote(k_nearest_labels)
        
        return predictions

    def set_params(self, k=None, distance_metric=None, batch_size=None):
        if k is not None:
            self.k = k
        if distance_metric is not None:
            self.distance_metric = distance_metric
        if batch_size is not None:
            self.batch_size = batch_size

class Metrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true, y_pred):
        true_positives = np.sum((y_true == y_pred) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / predicted_positives if predicted_positives > 0 else 0
    
    @staticmethod
    def recall(y_true, y_pred):
        true_positives = np.sum((y_true == y_pred) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / actual_positives if actual_positives > 0 else 0
    
    @staticmethod
    def f1_score(y_true, y_pred):
        precision = Metrics.precision(y_true, y_pred)
        recall = Metrics.recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0