import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class MLP_multilabel:
    def __init__(self, hidden_neurons, num_hid_layers, epochs, learning_rate=0.01, activation='tanh', optimizer='batch_gd', batch_size=None):
        self.hidden_layer_sizes = hidden_neurons
        self.num_hidden = num_hid_layers
        self.learning_rate = learning_rate
        self.activation = activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        if(optimizer == "sgd"):
            self.batch_size = 1
        elif(optimizer == "batch_gd"):
            self.batch_size = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid_derivative(self, activations):
        return activations * (1 - activations)

    def tanh_derivative(self, activations):
        return 1 - np.power(activations, 2)

    def relu_derivative(self, activations):
        return np.where(activations > 0, 1, 0)

    def activation_function(self, weighted_sums):
        if self.activation == "sigmoid":
            return self.sigmoid(weighted_sums)
        elif self.activation == "tanh":
            return self.tanh(weighted_sums)
        elif self.activation == "relu":
            return self.relu(weighted_sums)

    def activation_derivative(self, activations):
        if self.activation == "sigmoid":
            return self.sigmoid_derivative(activations)
        elif self.activation == "tanh":
            return self.tanh_derivative(activations)
        elif self.activation == "relu":
            return self.relu_derivative(activations)

    def forward_propagation(self, X):
        activations = [X]
        weighted_sums = []
        for i in range(len(self.weights) - 1):
            arr = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            weighted_sums.append(arr)
            activations.append(self.activation_function(arr))
        arr = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        activations.append(self.sigmoid(arr))
        weighted_sums.append(arr)
        return activations, weighted_sums

    def backward_propagation(self, X, y, activations, weighted_sums):
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]
        delta = (activations[-1] - y)
        gradients_w[-1] = np.dot(activations[-2].T, delta)
        gradients_b[-1] = np.sum(delta, axis=0, keepdims=True)
        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(delta, self.weights[i + 1].T) * self.activation_derivative(activations[i + 1])
            gradients_w[i] = np.dot(activations[i].T, delta)
            gradients_b[i] = np.sum(delta, axis=0, keepdims=True)
        for i in range(len(self.weights)):
            self.weights[i] -= (self.learning_rate * gradients_w[i])/self.batch_size
            self.biases[i] -= (self.learning_rate * gradients_b[i])/self.batch_size

    def cross_entropy_loss(self,Y_true, Y_pred):
        loss = -np.sum(Y_true* np.log(Y_pred)) / Y_true.shape[0]
        return loss
    
    def train(self, X, Y):
        X = X.to_numpy()
        self.input_size = X.shape[1]
        self.output_size = len(set_of_labels)
        self.Y_min = Y.min()
        layer_sizes = np.hstack((np.array([self.input_size]), self.hidden_layer_sizes, np.array([self.output_size])))
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) / np.sqrt(layer_sizes[i + 1]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
        if self.batch_size is None or self.batch_size>X.shape[0]:
            self.batch_size = X.shape[0]
        Y_true = Y.values
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y_true = Y_true[indices]
        for j in range(self.epochs):
            loss_arr = []
            for i in range(0,X.shape[0],self.batch_size):
                X_batch = X[i:i + self.batch_size]
                Y_batch = Y_true[i:i + self.batch_size]
                activations, weighted_sums = self.forward_propagation(X_batch)
                self.backward_propagation(X_batch, Y_batch, activations, weighted_sums)
                loss_arr.append(self.cross_entropy_loss(Y_batch,activations[-1]))
            self.train_loss  = np.mean(loss_arr)

    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        predicted_classes = (activations[-1] >= 0.5).astype(int)
        return predicted_classes


# Load and preview the data
data_path = '../../data/external/advertisement.csv'
data_adv = pd.read_csv(data_path)

labels =set()
for i in range(len(data_adv)):
    data_labels = (data_adv.iloc[i]["labels"]).split()
    for j in data_labels:
        labels.add(j)
set_of_labels= np.array(list(sorted(labels)))
df_encoded = data_adv.__deepcopy__()
label_encoder = LabelEncoder()
columns=["gender","education","city","occupation","most bought item"]
for i in columns:
    df_encoded[i] = label_encoder.fit_transform(df_encoded[i])
X_df  =df_encoded.drop("labels",axis="columns")
Y_df_1 = data_adv["labels"]
Y_df  =pd.DataFrame()
for label in set_of_labels:
    Y_df[label] = Y_df_1.apply(lambda ele:1 if label in ele else 0)
    

from sklearn.preprocessing import StandardScaler
X_train , X_test, Y_train, Y_test = train_test_split(X_df,Y_df,test_size=0.25,random_state=42)
X_test, X_val , Y_test , Y_val = train_test_split(X_test,Y_test,test_size=0.5,random_state=42)
scaler  = StandardScaler()
scaler.fit(X_df)
X_train = pd.DataFrame(scaler.transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))
X_val = pd.DataFrame(scaler.transform(X_val))


def hamming_score(y_true,y_pred):
    if(isinstance(y_true,pd.DataFrame)):
        y_true= y_true.to_numpy()
    if(isinstance(y_pred,pd.DataFrame)):
        y_pred = y_pred.to_numpy()
    x=np.array([])
    x2=np.array([])
    for i in range(len(y_true)):
        x = np.hstack((x,y_true[i]))
        x2 = np.hstack((x2,y_pred[i]))
    return accuracy_score(x,x2)
def f1_score_hamming(y_true,y_pred):
    if(isinstance(y_true,pd.DataFrame)):
        y_true= y_true.to_numpy()
    if(isinstance(y_pred,pd.DataFrame)):
        y_pred = y_pred.to_numpy()
    x=np.array([])
    x2=np.array([])
    for i in range(len(y_true)):
        x = np.hstack((x,y_true[i]))
        x2 = np.hstack((x2,y_pred[i]))
    return f1_score(x,x2,average='macro',zero_division=0)
def precision_hamming(y_true,y_pred):
    if(isinstance(y_true,pd.DataFrame)):
        y_true= y_true.to_numpy()
    if(isinstance(y_pred,pd.DataFrame)):
        y_pred = y_pred.to_numpy()
    x=np.array([])
    x2=np.array([])
    for i in range(len(y_true)):
        x = np.hstack((x,y_true[i]))
        x2 = np.hstack((x2,y_pred[i]))
    return precision_score(x,x2,average='macro',zero_division=0)
def recall_hamming(y_true,y_pred):
    if(isinstance(y_true,pd.DataFrame)):
        y_true= y_true.to_numpy()
    if(isinstance(y_pred,pd.DataFrame)):
        y_pred = y_pred.to_numpy()
    x=np.array([])
    x2=np.array([])
    for i in range(len(y_true)):
        x = np.hstack((x,y_true[i]))
        x2 = np.hstack((x2,y_pred[i]))
    return recall_score(x,x2,average='macro',zero_division=0)


import wandb

def run_wandb_experiments(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    wandb.login()

    optimizers = ["sgd", "batch_gd", "mini_gd"]
    activation_funcs = ["relu", "tanh", "sigmoid"]
    batch_sizes = [8, 16, 32]
    epochs = [100, 1000, 1500, 3000]
    learning_rates = [0.001, 0.002, 0.004]

    for optimizer in optimizers:
        for activation in activation_funcs:
            with wandb.init(project="MLP-Classifier-MultiLabel", name=f"Optimizer={optimizer}, Activation={activation}") as run:
                config = wandb.config
                config.optimizer = optimizer
                config.activation = activation

                for rate in learning_rates:
                    for num_epoch in epochs:
                        if optimizer == "mini_gd":
                            for batch in batch_sizes:
                                model = MLP_multilabel(hidden_neurons=[10], num_hid_layers=1, epochs=num_epoch,
                                                       learning_rate=rate, activation=activation,
                                                       optimizer=optimizer, batch_size=batch)
                                run_experiment(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, batch)
                        else:
                            model = MLP_multilabel(hidden_neurons=[10], num_hid_layers=1, epochs=num_epoch,
                                                   learning_rate=rate, activation=activation, optimizer=optimizer)
                            run_experiment(model, X_train, Y_train, X_val, Y_val, X_test, Y_test)

def run_experiment(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, batch_size=None):
    model.train(X_train, Y_train)
    Y_pred_val = model.predict(X_val)
    accuracy_val = hamming_score(Y_val, Y_pred_val)
    Y_pred_test = model.predict(X_test)
    accuracy_test = hamming_score(Y_test, Y_pred_test)

    log_data = {
        "num_epochs": model.epochs,
        "learning_rate": model.learning_rate,
        "train_loss": model.train_loss,
        "acc_val": accuracy_val,
        "acc_test": accuracy_test
    }
    if batch_size:
        log_data["batch_size"] = batch_size

    wandb.log(log_data)
    
run_wandb_experiments(X_train, Y_train, X_val, Y_val, X_test, Y_test)

