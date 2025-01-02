class Metrics:
    
    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    
    def precision(y_true, y_pred):
        true_positives = np.sum((y_true == y_pred) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / predicted_positives if predicted_positives > 0 else 0

    
    def recall(y_true, y_pred):
        true_positives = np.sum((y_true == y_pred) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / actual_positives if actual_positives > 0 else 0
    
    
    def f1_score(y_true, y_pred):
        precision = Metrics.precision(y_true, y_pred)
        recall = Metrics.recall(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0