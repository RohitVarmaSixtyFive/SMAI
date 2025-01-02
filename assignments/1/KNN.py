import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from sklearn.neighbors import KNeighborsClassifier

# Add this path if necessary
sys.path.append(os.path.abspath('../../models/knn'))
from knn import KNNClassifier
from knn import Metrics

def load_data():
    data_path = os.path.abspath(os.path.join("1", "..", "..", "..", "data", "external", 'spotify.csv'))
    return pd.read_csv(data_path)

def impute_missing_values(df):
    for col in df.columns:
        if df[col].dtype.name != 'object':
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def custom_label_encoder(series):
    unique_vals = series.unique()
    val_to_int = {val: idx for idx, val in enumerate(unique_vals)}
    encoded = series.map(val_to_int)
    return encoded, val_to_int

def label_encode_columns(df, columns):
    label_encoders = {}
    for col in columns:
        df[col], encoder = custom_label_encoder(df[col])
        label_encoders[col] = encoder
    return df, label_encoders

def drop_unnecessary_columns(df):
    df.drop(columns=['track_id', 'track_name', 'Unnamed: 0', 'artists', 'album_name'], inplace=True, errors='ignore')
    return df

def encode_target_variable(df):
    df['track_genre'], genre_encoder = custom_label_encoder(df['track_genre'])
    return df, genre_encoder

def train_test_val_split(df, train_size=0.8, test_size=0.1):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(df))
    train_end = int(train_size * len(df))
    test_end = int(test_size * len(df)) + train_end
    
    train_indices = shuffled_indices[:train_end]
    test_indices = shuffled_indices[train_end:test_end]
    val_indices = shuffled_indices[test_end:]
    
    return df.iloc[train_indices], df.iloc[test_indices], df.iloc[val_indices]

def standardize(X_train, X_test, X_val, categorical_columns):
    num_columns = X_train.columns.difference(categorical_columns)
    mean = X_train[num_columns].mean(axis=0)
    std = X_train[num_columns].std(axis=0)
    
    X_train[num_columns] = (X_train[num_columns] - mean) / std
    X_test[num_columns] = (X_test[num_columns] - mean) / std
    X_val[num_columns] = (X_val[num_columns] - mean) / std
    
    return X_train, X_test, X_val

def tune_hyperparameters(X_train, y_train, X_val, y_val):
    k_values = [11,13,15,17, 19, 21, 23, 25]
    distance_metrics = ['euclidean', 'manhattan','cosine']
    results = []

    for k in k_values:
        for distance_metric in distance_metrics:
            knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
            knn.fit(X_train, y_train)
            y_pred_val = knn.predict(X_val)
            accuracy = Metrics.accuracy(y_val, y_pred_val)
            results.append((k, distance_metric, accuracy))

    results.sort(key=lambda x: x[2], reverse=True)
    return results

def plot_results(results, selected_metric='euclidean'):
    ks = [k for k, metric, _ in results if metric == selected_metric]
    accuracies = [accuracy for k, metric, accuracy in results if metric == selected_metric]

    plt.plot(ks, accuracies, marker='o')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs k (distance_metric={selected_metric})')
    plt.grid(True)
    plt.show()

def main():
    df = load_data()
    df = impute_missing_values(df)
    df, label_encoders = label_encode_columns(df, ['explicit'])
    df = drop_unnecessary_columns(df)
    df, genre_encoder = encode_target_variable(df)
    
    df_train, df_test, df_val = train_test_val_split(df)
    
    X_train = df_train.drop(columns=['track_genre'])
    X_test = df_test.drop(columns=['track_genre'])
    X_val = df_val.drop(columns=['track_genre'])
    
    y_train = df_train['track_genre']
    y_test = df_test['track_genre']
    y_val = df_val['track_genre']
    
    X_train_scaled, X_test_scaled, X_val_scaled = standardize(X_train, X_test, X_val, ['explicit'])
    
    X_train_np = X_train_scaled.values
    X_test_np = X_test_scaled.values
    X_val_np = X_val_scaled.values
    y_train_np = y_train.values
    y_val_np = y_val.values
    
    results = tune_hyperparameters(X_train_np, y_train_np, X_val_np, y_val_np)
    
    print("Top 10 (k, distance_metric) pairs by validation accuracy:")
    for i, (k, distance_metric, accuracy) in enumerate(results[:10]):
        print(f"{i+1}. k={k}, distance_metric={distance_metric}, accuracy={accuracy:.4f}")
    
    plot_results(results)

if __name__ == "__main__":
    main()
