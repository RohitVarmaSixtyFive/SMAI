import sys
import os 
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))  
import matplotlib.pyplot as plt
from models.PCA.PCA import PCA
from models.Kmeans.Kmeans import Kmeans
from models.GMM.GMM import GMM
import numpy as np

data_path = os.path.abspath(os.path.join("2","..", "..","..", "data", "external","word-embeddings.feather"))
class seven_header:
    def __init__(self):
        df = pd.read_feather(data_path)  
        string_array = np.array(df['vit'].tolist())  

        words = df['words'].tolist() 

        print(string_array.shape)

        pca = PCA(n_components=4)
        pca.fit(string_array) 

        reduced_data = pca.transform(string_array)

        k = 6
        kmeans = Kmeans(k=k)
        kmeans.fit(string_array)

        final_cluster_assignments = kmeans.labels

        clustered_words = {i: [] for i in range(k)} 
        for word, label in zip(words, final_cluster_assignments):
            clustered_words[label].append(word)

        for cluster, words_in_cluster in clustered_words.items():
            print(f"Cluster {cluster}:")
            print(", ".join(words_in_cluster)) 
            

        k = 7
        kmeans = Kmeans(k=k)
        kmeans.fit(reduced_data)

        final_cluster_assignments = kmeans.labels

        clustered_words = {i: [] for i in range(k)} 
        for word, label in zip(words, final_cluster_assignments):
            clustered_words[label].append(word)

        for cluster, words_in_cluster in clustered_words.items():
            print(f"Cluster {cluster}:")
            print(", ".join(words_in_cluster)) 
            
        k = 1
        kmeans = GMM(k=k)
        kmeans.fit(reduced_data)

        final_cluster_assignments = kmeans.predict(reduced_data)

        clustered_words = {i: [] for i in range(k)} 
        for word, label in zip(words, final_cluster_assignments):
            clustered_words[label].append(word)

        for cluster, words_in_cluster in clustered_words.items():
            print(f"Cluster {cluster}:")
            print(", ".join(words_in_cluster)) 
            
        k = 3
        kmeans = GMM(k=k)
        kmeans.fit(reduced_data)

        final_cluster_assignments = kmeans.predict(reduced_data)

        clustered_words = {i: [] for i in range(k)} 
        for word, label in zip(words, final_cluster_assignments):
            clustered_words[label].append(word)

        for cluster, words_in_cluster in clustered_words.items():
            print(f"Cluster {cluster}:")
            print(", ".join(words_in_cluster)) 
            
        k = 3
        kmeans = GMM(k=k)
        kmeans.fit(reduced_data)

        final_cluster_assignments = kmeans.predict(reduced_data)

        clustered_words = {i: [] for i in range(k)} 
        for word, label in zip(words, final_cluster_assignments):
            clustered_words[label].append(word)

        for cluster, words_in_cluster in clustered_words.items():
            print(f"Cluster {cluster}:")
            print(", ".join(words_in_cluster)) 
            
if __name__ == "__main__":
    seven_header()