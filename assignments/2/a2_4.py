import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

class GMM_header:

    def __init__(self):
        # Load the data paths
        self.data_path = os.path.abspath(os.path.join("2", "..", "..", "..", "data", "external", "word-embeddings.feather"))
        self.data_path2 = os.path.abspath(os.path.join("2", "..", "..", "..", "data", "external", "2d_clustering.csv"))
        
        # Load the datasets
        self.df_vis = pd.read_csv(self.data_path2)
        self.df = pd.read_feather(self.data_path)

        # Initialize the string array from embeddings
        self.string_array = np.array(self.df['vit'].tolist())
        print(self.string_array.shape)
        
        self.plot_gmm_clustering()
        self.plot_aic_bic()
        self.plot_aic_bic_for_embeddings()


    def plot_gmm_clustering(self):
        df_analysis = self.df_vis.copy()
        df_analysis = df_analysis.drop(columns=['color'])
        visualization = df_analysis[['x', 'y']].to_numpy()

        # Perform Gaussian Mixture Clustering
        n_clusters = 5
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(visualization)
        df_analysis['cluster'] = gmm.predict(visualization)

        # Plotting GMM clustering
        plt.figure(figsize=(10, 10))
        plt.scatter(df_analysis['x'], df_analysis['y'], c=df_analysis['cluster'], cmap='viridis')
        plt.title('GMM Clustering')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('plots/4_toydata_gmm-clustering.png')
        plt.show()

    def plot_aic_bic(self):
        df_analysis = self.df_vis.copy()
        df_analysis = df_analysis.drop(columns=['color'])
        visualization = df_analysis[['x', 'y']].to_numpy()

        AIC = []
        BIC = []
        k_range = range(1, 20)
        n_samples, n_features = visualization.shape

        # Calculate AIC and BIC for each number of clusters
        for i in k_range:
            gmm = GaussianMixture(n_components=i, covariance_type='full', n_init=10, tol=1e-3)
            gmm.fit(visualization)
            AIC.append(gmm.aic(visualization))
            BIC.append(gmm.bic(visualization))

        # Plot AIC and BIC
        plt.figure(figsize=(10, 7))
        plt.plot(k_range, AIC, label='AIC', marker='o')
        plt.plot(k_range, BIC, label='BIC', marker='s')
        plt.title('AIC and BIC Methods for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Score')
        plt.xticks(k_range)
        plt.grid(True)
        plt.legend()
        plt.savefig('plots/4_toydata_gmm-aic-bic.png')
        plt.show()

    def plot_aic_bic_for_embeddings(self):
        AIC = []
        BIC = []
        k_range = range(1, 50)
        n_samples, n_features = self.string_array.shape

        # Calculate AIC and BIC for each number of clusters on word embeddings
        for i in k_range:
            gmm = GaussianMixture(n_components=i, covariance_type='full', n_init=10, tol=1e-3)
            gmm.fit(self.string_array)
            AIC.append(gmm.aic(self.string_array))
            BIC.append(gmm.bic(self.string_array))

        # Plot AIC and BIC
        plt.figure(figsize=(12, 8))
        plt.plot(k_range, AIC, label='AIC', marker='o')
        plt.plot(k_range, BIC, label='BIC', marker='s')
        plt.title('AIC and BIC Methods for Optimal K on Word Embeddings')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Score')
        plt.xticks(k_range)
        plt.grid(True)
        plt.legend()
        plt.savefig('plots/4_main_gmm-aic-bic.png')
        plt.show()


if __name__ == "__main__":
    GMM_header()
