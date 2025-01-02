import sys
import os 
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))  
import matplotlib.pyplot as plt
from models.PCA.PCA import PCA
from models.Kmeans.Kmeans import Kmeans
from models.GMM.GMM import GMM
import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import fcluster


data_path = os.path.abspath(os.path.join("2","..", "..","..", "data", "external","word-embeddings.feather"))


class eight_header:
    def __init__(self):
        df = pd.read_feather(data_path)  
        string_array = np.array(df['vit'].tolist())  

        words = df['words'].tolist() 

        print(string_array.shape)


        # Define the linkage methods and distance metrics
        linkage_methods = ['single', 'complete', 'average', 'ward']
        distance_metrics = ['euclidean', 'cosine', 'cityblock']  # Euclidean, Cosine, Manhattan (Cityblock)

        X = string_array  # Your data

        plt.figure(figsize=(20, 15))

        plot_num = 1
        # Iterate over all combinations of linkage methods and distance metrics
        for method in linkage_methods:
            for metric in distance_metrics:
                # Skip non-Euclidean metrics for the 'ward' method
                # if method == 'ward' and metric != 'euclidean':
                #     continue

                # Compute the linkage matrix
                dist_matrix = pdist(X, metric=metric)
                linkage_matrix = linkage(dist_matrix, method=method)

                # Create a subplot for each combination
                plt.subplot(len(linkage_methods), len(distance_metrics), plot_num)
                dendrogram(linkage_matrix)
                plt.title(f'{method} linkage, {metric} metric')
                plt.xlabel('Word Index')
                plt.ylabel('Distance')
                
                plot_num += 1

        plt.tight_layout()
        plt.savefig('plots/8_dendrogram_combinations.png')
        plt.show()


        kbest1 = 6  
        kbest2 = 3 

        Z_best = linkage(X, method='ward', metric='euclidean')

        clusters_kbest1 = fcluster(Z_best, t=kbest1, criterion='maxclust')

        clusters_kbest2 = fcluster(Z_best, t=kbest2, criterion='maxclust')

        print(clusters_kbest1)

        clustered_words = {i: [] for i in range(1, kbest1 + 1)}  # Labels start from 1, so range should go from 1 to kbest1

        # Assign words to corresponding clusters
        for word, label in zip(words, clusters_kbest1):
            clustered_words[label].append(word)

        # Print results
        for cluster, words_in_cluster in clustered_words.items():
            print(f"Cluster {cluster}:")
            print(", ".join(words_in_cluster))
            
            
        clustered_words = {i: [] for i in range(1, kbest2 + 1)}  # Labels start from 1, so range should go from 1 to kbest1

        # Assign words to corresponding clusters
        for word, label in zip(words, clusters_kbest2):
            clustered_words[label].append(word)

        # Print results
        for cluster, words_in_cluster in clustered_words.items():
            print(f"Cluster {cluster}:")
            print(", ".join(words_in_cluster))
            
if __name__ == '__main__':
    eight_header()