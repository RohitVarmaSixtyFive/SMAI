import numpy as np  
  
class LinearRegression:  
   def __init__(self, learning_rate=0.01, degree=1, regularization=None, lamda=0, max_iter=1000,tolerance=1e-5):  
      self.learning_rate = learning_rate  
      self.regularization = regularization  
      self.lamda = lamda  
      self.max_iter = max_iter  
      self.degree = degree  
      self.weights = None  
      self.tolerance = 1e-5
  
   def initialize_weights(self, n_features):  
      self.weights = np.zeros(n_features + 1)
  
   def add_bias(self, X):  
      return np.hstack((np.ones((X.shape[0], 1)), X))  
  
   def polynomial_features(self, X):  
      return np.hstack([X ** i for i in range(1, self.degree + 1)])  
  
   def compute_gradient(self, X, y, y_pred):  
      n_samples = X.shape[0]  
      error = y - y_pred  
      dcost_dw = -2 * np.dot(X.T, error) / n_samples
      gradient = dcost_dw  
  
      if self.regularization == 'L1':  
        gradient[1:] += self.lamda * np.sign(self.weights[1:]) 
      elif self.regularization == 'L2':  
        gradient[1:] += 2 * self.lamda * self.weights[1:]
  
      return gradient  
  
   def gradient_descent_step(self, X, y):  
      y_pred = np.dot(X, self.weights)  
      gradient = self.compute_gradient(X, y, y_pred)  
      self.weights -= self.learning_rate * gradient  
      return y_pred
  
   def fit(self, X, y):  
      if self.degree > 1:  #stack colsss
        X = self.polynomial_features(X)
      n_samples, n_features = X.shape  
      X = self.add_bias(X)  
      self.initialize_weights(n_features)  
  
      previous_loss = float('inf')
        
      for i in range(self.max_iter):
         y_pred = self.gradient_descent_step(X, y)
         current_loss = self.mse(y, y_pred)
         
         if abs(previous_loss - current_loss) < self.tolerance:
               # print(f"Early stopping at iteration {i+1}")
               break
         previous_loss = current_loss
  
   def predict(self, X):  
      if self.degree > 1:  
        X = self.polynomial_features(X)  
      X = self.add_bias(X)  
      return np.dot(X, self.weights)  
  
   def mse(self, y, y_pred):  
      return np.mean((y - y_pred) ** 2)  
  
   def std_dev(self, y, y_pred):  
      return np.std(y - y_pred)  
  
   def variance(self, y, y_pred):  
      return np.var(y - y_pred)  
  
   def get_weights(self):  
      return self.weights