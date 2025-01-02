import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))

DATA_PATH_LINREG = os.path.abspath(os.path.join("1", "..", "..", "..", "data", "external", 'linreg.csv'))
DATA_PATH_REG = os.path.abspath(os.path.join("1", "..", "..", "..", "data", "external", 'regularisation.csv'))

def load_data(data_path):
    return np.genfromtxt(data_path, delimiter=',', skip_header=1)

def shuffle_and_split_data(data):
    np.random.shuffle(data)
    train_size = int(0.8 * len(data))
    val_size = int(0.1 * len(data))
    
    X_train, y_train = data[:train_size, 0], data[:train_size, 1]
    X_val, y_val = data[train_size:train_size + val_size, 0], data[train_size:train_size + val_size, 1]
    X_test, y_test = data[train_size + val_size:, 0], data[train_size + val_size:, 1]
    
    return X_train, X_test, X_val, y_train, y_test, y_val

def plot_data_splits(X_train, y_train, X_val, y_val, X_test, y_test):
    plt.scatter(X_train, y_train, label='Training Set', color='black')
    plt.scatter(X_val, y_val, label='Validation Set', color='yellow')
    plt.scatter(X_test, y_test, label='Test Set', color='red')
    plt.xlabel('Feature')
    plt.ylabel('Target Variable')
    plt.title('Data Splits')
    plt.legend()
    plt.show()

def plot_polynomial_fit(X_train, y_train, model, degree):
    X_fit = np.linspace(min(X_train), max(X_train), 100).reshape(-1, 1)
    y_fit = model.predict(X_fit)
    
    plt.scatter(X_train, y_train, color='blue', label='Training data')
    plt.plot(X_fit, y_fit, color='red', label=f'Polynomial degree {degree}')
    plt.xlabel('Independent Variable')
    plt.ylabel('Dependent Variable')
    plt.title('Polynomial Regression Fit')
    plt.legend()
    plt.show()

def create_animation(X, y, model, filename='linear_regression_animation.gif'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    line1, = ax1.plot([], [], 'green', label='Fitted')
    scatter1 = ax1.scatter(X, y, color='yellow', label='Observations')
    
    line2, = ax2.plot([], [], 'green', label='Fitted')
    scatter2 = ax2.scatter(X, y, color='yellow', label='Observations')
    residual_lines = [ax2.plot([], [], 'r-')[0] for _ in range(len(X))]
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    ax1.legend()
    ax2.legend()
    
    text1 = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, verticalalignment='top')
    text2 = ax2.text(0.05, 0.95, '', transform=ax2.transAxes, verticalalignment='top')

    def init():
        ax1.set_xlim(X.min(), X.max())
        ax1.set_ylim(y.min(), y.max())
        ax2.set_xlim(X.min(), X.max())
        ax2.set_ylim(y.min(), y.max())
        return line1, line2, scatter1, scatter2, text1, text2, *residual_lines

    def update(frame):
        weights = model.history[frame]['weights']
        bias = model.history[frame]['bias']
        epoch = model.history[frame]['epoch']
        loss = model.history[frame]['loss']
        
        y_pred = np.dot(X, weights) + bias
        
        line1.set_data(X.flatten(), y_pred)
        line2.set_data(X.flatten(), y_pred)
        
        text1.set_text(f'epoch = {epoch}, β = {weights[0]:.3f}, b = {bias:.5f}')
        text2.set_text(f'loss = {loss:.3f}')
        
        for i, residual_line in enumerate(residual_lines):
            residual_line.set_data([X[i], X[i]], [y[i], y_pred[i]])
        
        return line1, line2, scatter1, scatter2, text1, text2, *residual_lines

    anim = FuncAnimation(fig, update, frames=len(model.history), init_func=init, blit=True)
    anim.save(filename, writer=PillowWriter(fps=10))

def main():
    
    import KNN
    
    KNN.main()
        
    data_linreg = load_data(DATA_PATH_LINREG)
    X_train, X_test, X_val, y_train, y_test, y_val = shuffle_and_split_data(data_linreg)
    X_train = X_train.reshape(-1, 1)
    X_val = X_val.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    plot_data_splits(X_train, y_train, X_val, y_val, X_test, y_test)

    from models.linear_regression.linear_regression import LinearRegression
    degree = 1
    model = LinearRegression(degree=degree, learning_rate=0.01, max_iter=10000)
    model.fit(X_train, y_train)
    plot_polynomial_fit(X_train, y_train, model, degree)

    degrees = np.linspace(1, 20, 20).astype(int)
    results = {}

    for k in degrees:
        model = LinearRegression(degree=k, learning_rate=0.01, max_iter=10000)
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_mse = model.mse(y_train, y_pred_train)
        test_mse = model.mse(y_test, y_pred_test)
        
        train_std_dev = model.std_dev(y_train, y_pred_train)
        test_std_dev = model.std_dev(y_test, y_pred_test)
        
        train_variance = model.variance(y_train, y_pred_train)
        test_variance = model.variance(y_test, y_pred_test)
        
        results[k] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_std_dev': train_std_dev,
            'test_std_dev': test_std_dev,
            'train_variance': train_variance,
            'test_variance': test_variance
        }
    
    best_k = min(results, key=lambda k: results[k]['test_mse'])
    print(f"Best degree (k) with minimum test MSE: {best_k}")

    weights_file_path = 'best_model_weights.txt'
    best_model = LinearRegression(degree=best_k, learning_rate=0.01, max_iter=10000)
    best_model.fit(X_train, y_train)
    np.savetxt(weights_file_path, best_model.get_weights())
    print(f"Best model weights saved to {weights_file_path}")

    loaded_weights = np.loadtxt(weights_file_path)
    loaded_model = LinearRegression(degree=best_k, learning_rate=0.01, max_iter=10000)
    loaded_model.weights = loaded_weights
    y_pred_test = loaded_model.predict(X_test)

    # model_animation = FuncAnimation(fig, update, frames=len(model.history), init_func=init, blit=True)
    # model_animation.fit(X_train, y_train)
    # create_animation(X_train, y_train, model_animation)

    data_reg = load_data(DATA_PATH_REG)
    X_train, X_test, X_val, y_train, y_test, y_val = shuffle_and_split_data(data_reg)
    X_train = X_train.reshape(-1, 1)
    X_val = X_val.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    
    plot_data_splits(X_train, y_train, X_val, y_val, X_test, y_test)

    degrees = np.linspace(1, 30, 30).astype(int)
    lambdas = [0.1, 1, 10, 20, 50]
    regularizations = ['L1', 'L2']

    results = {}
    for k in degrees:
        for lamda in lambdas:
            for regularization in regularizations:
                model = LinearRegression(degree=k, learning_rate=0.01, max_iter=10000, lamda=lamda, regularization=regularization)
                model.fit(X_train, y_train)
                
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                train_mse = model.mse(y_train, y_pred_train)
                test_mse = model.mse(y_test, y_pred_test)
                
                train_std_dev = model.std_dev(y_train, y_pred_train)
                test_std_dev = model.std_dev(y_test, y_pred_test)
                
                train_variance = model.variance(y_train, y_pred_train)
                test_variance = model.variance(y_test, y_pred_test)
                
                results[(k, lamda, regularization)] = {
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_std_dev': train_std_dev,
                    'test_std_dev': test_std_dev,
                    'train_variance': train_variance,
                    'test_variance': test_variance
                }
    
    best_params = min(results, key=lambda params: results[params]['test_mse'])
    print(f"Best parameters (degree, lambda, regularization) with minimum test MSE: {best_params}")

if __name__ == "__main__":
    main()
