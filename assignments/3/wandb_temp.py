from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import numpy as np
import pandas as pd
import os
import sys
import wandb

data_path = '../../data/external/WineQT.csv'
df = pd.read_csv(data_path)

import matplotlib.pyplot as plt
import numpy as np

wine_quality_labels = df['quality'].values

plt.figure(figsize=(8,6))

plt.hist(wine_quality_labels, bins=np.arange(2.5, 9.5, 1), rwidth=0.8)

plt.title('Distribution of Wine Quality Labels')
plt.xlabel('Wine Quality')
plt.ylabel('Frequency') 
plt.show()

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))  
from models.MLP.MLP import MLP_SingleLabelClassifier

import warnings
warnings.filterwarnings('ignore')

df.fillna(0, inplace=True)
df = df.sample(frac=1, random_state=42)

X = df.drop('quality', axis=1)
y = df['quality']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y) ## sending in as 0 1 2 3 4 5

scaler = StandardScaler()
X = scaler.fit_transform(X)

def objective():
    run = wandb.init(project="MLP_Hyperparameter_Tuning") 
    config = wandb.config  

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
    
    model = MLP_SingleLabelClassifier(
        input_size=X_train.shape[1],
        hidden_layers=config.hidden_layers,
        output_size=len(np.unique(y)),
        learning_rate=config.learning_rate,
        activation=config.activation,
        optimizer=config.optimizer,
        batch_size=config.batch_size,
        epochs=config.epochs
    )
    
    model.fit(X_train, y_train, validation=True, X_val=X_val, y_val=y_val)

    y_val_pred = model.predict(X_val)
    final_val_acc = accuracy_score(y_val, y_val_pred)
    final_f1 = f1_score(y_val, y_val_pred, average='weighted')
    final_precision = precision_score(y_val, y_val_pred, average='weighted')
    final_recall = recall_score(y_val, y_val_pred, average='weighted')

    wandb.log({
        'final_val_acc': final_val_acc,
        'final_f1_score': final_f1,
        'final_precision': final_precision,
        'final_recall': final_recall
    })

    if final_val_acc >= wandb.run.summary.get('best_val_acc', 0):
        wandb.run.summary['best_val_acc'] = final_val_acc
        # wandb.save('best_model.pkl')

    return final_val_acc

sweep_config = {
    'method': 'grid', 
    'metric': {'name': 'final_val_acc', 'goal': 'maximize'},
    'parameters': {
        'hidden_layers': {'values': [[64], [32, 32], [32, 64], [32, 32, 32]]},
        'learning_rate': {'values': [0.001, 0.01]},
        'activation': {'values': ['relu', 'sigmoid', 'linear', 'tanh']},
        'optimizer': {'values': ['sgd', 'mini_batch_gd', 'batch_gd']},
        'batch_size': {'values': [16, 32]},
        'epochs': {'values': [100, 200]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="MLP_SingleLabelClassifier")
wandb.agent(sweep_id, function=objective)