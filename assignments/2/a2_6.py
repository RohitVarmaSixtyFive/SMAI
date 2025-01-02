import sys
import os 
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))  
import matplotlib.pyplot as plt
from models.PCA.PCA import PCA
from models.Kmeans.Kmeans import Kmeans
import numpy as np



import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from models.PCA.PCA import PCA  
from models.GMM.GMM import GMM
from models.Kmeans.Kmeans import Kmeans


data_path = os.path.abspath(os.path.join("2","..", "..","..", "data", "external","word-embeddings.feather"))

class six_header:
    def __init__(self):
        df = pd.read_feather(data_path)  
        string_array = np.array(df['vit'].tolist())  

        words = df['words'].tolist() 

        print(string_array.shape)

        k2 = 3
        kmeans = Kmeans(k=k2)

        kmeans.fit(string_array)

        cluster_labels = kmeans.labels

        clustered_words = {i: [] for i in range(k2)}  
        for word, label in zip(words, cluster_labels):
            clustered_words[label].append(word)

        for cluster, words_in_cluster in clustered_words.items():
            print(f"Cluster {cluster}:")
            print(", ".join(words_in_cluster)) 


        pca = PCA(n_components=string_array.shape[1]) 
        pca.fit(string_array)

        explained_variance_ratio = pca.ret_importance()

        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(explained_variance_ratio)+1), explained_variance_ratio, marker='o', linestyle='--')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.grid(True)
        plt.savefig("6_big_scree_plot.png")
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, 10), explained_variance_ratio[1:10], marker='o', linestyle='--')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.grid(True)
        plt.savefig("6_small_scree_plot.png")
        plt.show()


        pca = PCA(n_components=4)
        pca.fit(string_array) 

        reduced_data = pca.transform(string_array)

        wcss = []
        k_range = range(1, 15)
        for k in k_range:
            kmeans = Kmeans(k=k)
            kmeans.fit(reduced_data)
            wcss.append(kmeans.cost)

        plt.figure(figsize=(10, 7))
        plt.plot(k_range, wcss)
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('WCSS')
        plt.xticks(k_range)
        plt.grid(True)
        plt.savefig('plots/6_main_kmeans_data_elbow.png')
        plt.show()


        optimal_k = 7
        kmeans = Kmeans(k=optimal_k, random_state=42)
        kmeans.fit(reduced_data)

        final_cluster_assignments = kmeans.labels

        kmeans3 = optimal_k

        kmeans = Kmeans(k=kmeans3)
        kmeans.fit(reduced_data)

        cluster_labels = kmeans.labels

        clustered_words = {i: [] for i in range(optimal_k)} 
        for word, label in zip(words, cluster_labels):
            clustered_words[label].append(word)

        for cluster, words_in_cluster in clustered_words.items():
            print(f"Cluster {cluster}:")
            print(", ".join(words_in_cluster[:20])) 


        AIC = []
        BIC = []
        k_range = range(1, 10)

        n_samples, n_features = reduced_data.shape

        for i in k_range:
            gmm = GMM(k=i)
            gmm.fit(np.real(reduced_data))
            LL = gmm.getLikelihood(np.real(reduced_data))
            
            n_params = (i * n_features) + (i * n_features * (n_features + 1) // 2) + (i - 1)
            
            AIC.append(2 * n_params - 2 * LL)
            BIC.append(n_params * np.log(n_samples) - 2 * LL)


        plt.figure(figsize=(10, 7))
        plt.plot(k_range, AIC)
        plt.title('AIC method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('AIC')
        plt.xticks(k_range)
        plt.grid(True)
        plt.savefig('plots/4_toydata_gmm-aic.png')
        plt.show()    

        plt.figure(figsize=(10, 7))
        plt.plot(k_range, BIC)
        plt.title('BIC method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('BIC')
        plt.xticks(k_range)
        plt.grid(True)
        plt.savefig('plots/4_toydata_gmm-bic.png')
        plt.show()   

        optimal_k = 3
        gmm = GMM(k = optimal_k)

        gmm.fit(reduced_data)

        cluster_labels = gmm.predict(reduced_data)

        clustered_words = {i: [] for i in range(optimal_k)} 
        for word, label in zip(words, cluster_labels):
            clustered_words[label].append(word)

        for cluster, words_in_cluster in clustered_words.items():
            print(f"Cluster {cluster}:")
            print(", ".join(words_in_cluster)) 


if __name__ == "__main__":
    six_header()