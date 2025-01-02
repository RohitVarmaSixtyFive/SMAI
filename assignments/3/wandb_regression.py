from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import wandb
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))  
from models.MLP.MLP import MLP_Regressor

import warnings
warnings.filterwarnings('ignore')

data_path = '../../data/external/HousingData.csv'
df = pd.read_csv(data_path)

df = df.fillna(df.median())  

X = df.drop(columns=['MEDV'])
y = df['MEDV'].values.reshape(-1, 1)


def objective():
    run = wandb.init(project="MLP_Regressor_Hyperparameter_Tuning") 
    config = wandb.config  

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
    
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    
    model = MLP_Regressor(
        input_size=X_train.shape[1],
        hidden_layers=config.hidden_layers,
        output_size=1,
        learning_rate=config.learning_rate,
        activation=config.activation,
        optimizer=config.optimizer,
        batch_size=config.batch_size,
        epochs=config.epochs,
        wandb_logging=True
    )
    
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    y_val_pred = model.predict(X_val).flatten()
    final_mse = mean_squared_error(y_val, y_val_pred)
    final_mae = mean_absolute_error(y_val, y_val_pred)
    final_r2 = r2_score(y_val, y_val_pred)

    wandb.log({
        'final_mse': final_mse,
        'final_mae': final_mae,
        'final_r2': final_r2
    })

    if final_mse <= wandb.run.summary.get('best_val_mse', float('inf')):
        wandb.run.summary['best_val_mse'] = final_mse
        # wandb.save('best_model.pkl') # Optional model saving

    return final_mse

sweep_config = {
    'method': 'grid', 
    'metric': {'name': 'final_mse', 'goal': 'minimize'},
    'parameters': {
        'hidden_layers': {'values': [[64], [32, 32], [32, 64], [32, 32, 32]]},
        'learning_rate': {'values': [0.001, 0.01]},
        'activation': {'values': ['relu', 'sigmoid', 'linear', 'tanh']},
        'optimizer': {'values': ['sgd', 'mini_batch_gd', 'batch_gd']},
        'batch_size': {'values': [16, 32]},
        'epochs': {'values': [100, 200]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="MLP_Regressor")
wandb.agent(sweep_id, function=objective)
