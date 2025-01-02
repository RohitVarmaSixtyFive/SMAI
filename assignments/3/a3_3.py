import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
import wandb  
  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))  
# imports
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from models.MLP.MLP import MLP_Regressor

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from models.MLP.MLP import MLP_Reg

class third:
    def main():
        data_path = '../../data/external/HousingData.csv'
        df = pd.read_csv(data_path)

        df.describe()

        plt.figure(figsize=(10,6))  
        plt.hist(df['MEDV'], bins=50)  
        plt.title('Distribution of Median House Prices')  
        plt.xlabel('Median House Price ($1000s)')  
        plt.ylabel('Frequency')  
        plt.show()  
        
        df.dropna(inplace=True)

        X = df.drop('MEDV', axis=1)
        y = df['MEDV']



        sns.set(style="whitegrid", palette="pastel")

        num_columns = df.shape[1]  
        plots_per_row = 3  
        num_rows = (num_columns + plots_per_row - 1) // plots_per_row 

        plt.figure(figsize=(16, num_rows * 4))  

        for i, column in enumerate(df.columns):
            plt.subplot(num_rows, plots_per_row, i + 1)  
            sns.histplot(df[column], bins=20, kde=True, color='skyblue')  
            plt.title(f'Distribution of {column}', fontsize=14)
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.grid(True)

        plt.tight_layout()
        plt.show()



        data_path = '../../data/external/HousingData.csv'
        df = pd.read_csv(data_path)

        df = df.fillna(df.median())  

        X = df.drop(columns=['MEDV'])
        y = df['MEDV'].values.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # X_test_before = X_test.copy()

        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)

        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train)
        y_test = scaler_y.transform(y_test)

        mlp = MLP_Regressor(input_size=X_train.shape[1], hidden_layers=[32, 32], output_size=1,
                            learning_rate=0.01, activation='sigmoid', optimizer='sgd',
                            batch_size=32, epochs=200,  patience=10)

        mlp.fit(X_train, y_train, validation_split=0.2)

        y_train_pred = mlp.predict(X_train)
        y_test_pred = mlp.predict(X_test)

        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        # print(np.sum((y_test_pred - y_test)**2))
        # print(np.sum((y_test_pred - np.mean(y_test))**2))
        # print(mean_squared_error(y_test,np.mean(y_test)*np.ones(y_test.shape[0])))

        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"Training Data Metrics:\n  MSE: {train_mse}, MAE: {train_mae}, R2: {train_r2}\n")
        print(f"Test Data Metrics:\n  MSE: {test_mse}, MAE: {test_mae}, R2: {test_r2}\n")


        mse_values = (y_test - y_test_pred) ** 2

        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.set_xlabel('Index')
        ax1.set_ylabel('Predicted Values (y_pred)', color='tab:blue')
        ax1.plot(y_test_pred, label='y_pred', color='tab:blue', marker='o', linestyle='-', markersize=4)
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()  
        ax2.set_ylabel('MSE', color='tab:red')
        ax2.plot(mse_values, label='MSE', color='tab:red', marker='x', linestyle='-', markersize=4)
        ax2.tick_params(axis='y', labelcolor='tab:red')

        plt.title('Predicted Values vs. Index and MSE vs. Index')
        fig.tight_layout()  
        plt.show()

        print("y_test = ",y_test[1:10])
        print("y_test_pred = ",y_test_pred[1:10])

        print(mlp.check_gradients(X_train, y_train))



        data_path = '../../data/external/HousingData.csv'
        df = pd.read_csv(data_path)
        df = df.fillna(df.median())
        X = df.drop(columns=['MEDV'])
        y = df['MEDV']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        Y_train = scaler.fit_transform(y_train.values.reshape(-1, 1))
        Y_test = scaler.transform(y_test.values.reshape(-1, 1))

        mlp = MLP_Reg(hidden_neurons=[32, 32], num_hid_layers=2, epochs=200, 
                    learning_rate=0.01, activation='sigmoid', optimizer='sgd', batch_size=16)

        print("X shape = ",X_train.shape)
        print("Y shape = ",Y_train.shape)

        mlp.train(pd.DataFrame(X_train), pd.Series(y_train))

        y_train_pred = mlp.predict(X_train).flatten()
        y_test_pred = mlp.predict(X_test).flatten()

        test_mse = mean_squared_error(y_test, y_test_pred)

        mse_values = (y_test.values - y_test_pred) ** 2

        print(f"Test MSE: {test_mse}")
        print(f"Test MAE: {mean_absolute_error(y_test, y_test_pred)}")
        print(f"Test R2: {r2_score(y_test, y_test_pred)}")

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Actual Values (y_test)', color='tab:blue')
        ax1.plot(y_test.values, label='y_test', color='tab:blue', marker='o', linestyle='-', markersize=4)
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('MSE', color='tab:red')
        ax2.plot(mse_values, label='MSE', color='tab:red', marker='x', linestyle='-', markersize=4)
        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.suptitle('Actual Values (y_test) and MSE for Each Prediction')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        fig.tight_layout()
        plt.show()