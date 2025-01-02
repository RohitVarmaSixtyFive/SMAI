import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.colors as mcolors
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
from models.Kmeans.Kmeans import Kmeans

class Kmeans_header:
    def __init__(self):
        # Load data
        self.data_path = os.path.abspath(os.path.join("2", "..", "..", "..", "data", "external", "word-embeddings.feather"))
        self.data_path2 = os.path.abspath(os.path.join("2", "..", "..", "..", "data", "external", "2d_clustering.csv"))
        
        self.df_vis = pd.read_csv(self.data_path2)
        self.df = pd.read_feather(self.data_path)
        
        print(self.df_vis.head())
        
        self.string_array = np.array(self.df['vit'].tolist())
        print(self.string_array.shape)
        

        self.run_visualization_kmeans()


        self.run_embeddings_kmeans()
        
    def run_visualization_kmeans(self):

        df_analysis = self.df_vis.copy()
        df_analysis = df_analysis.drop(columns=['color'])
        visualization = df_analysis[['x', 'y']].to_numpy()


        n_clusters = 3
        kmeans = Kmeans(k=n_clusters)
        kmeans.fit(visualization)


        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(df_analysis['x'], df_analysis['y'], c='grey', cmap='viridis')
        centroid_scatter = ax.scatter([], [], c='red', marker='x', s=100)
        cmap = plt.get_cmap('viridis', n_clusters)
        norm = mcolors.Normalize(vmin=0, vmax=n_clusters-1)

        def update(frame):
            if frame < len(kmeans.history_labels):
                labels = kmeans.history_labels[frame]
                centroids = kmeans.history_centroids[frame]
                scatter.set_color(cmap(norm(labels)))
                centroid_scatter.set_offsets(centroids)
                ax.set_title(f"Iteration {frame + 1}")
            return scatter, centroid_scatter

        total_frames = len(kmeans.history_labels)
        ani = FuncAnimation(fig, update, frames=total_frames, interval=1000, repeat=False)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(scatter, label='Cluster', ticks=range(n_clusters))
        plt.tight_layout()

        writer = PillowWriter(fps=1, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('plots/kmeans_clustering.gif', writer=writer)
        plt.close(fig)


        self.plot_elbow_method(visualization, '3_visualization_data_elbow.png')

    def run_embeddings_kmeans(self):

        n_clusters = 3
        kmeans = Kmeans(k=n_clusters)
        kmeans.fit(self.string_array)
        
        self.plot_elbow_method(self.string_array, '3_main_kmeans_data_elbow.png', max_k=15)

        optimal_k = 5  
        kmeans = Kmeans(k=optimal_k, random_state=42)
        kmeans.fit(self.string_array)
        final_cluster_assignments = kmeans.labels
        print(f"Final Cluster Assignments: {final_cluster_assignments}")

    def plot_elbow_method(self, data, plot_name, max_k=10):
        wcss = []
        k_range = range(1, max_k + 1)
        for k in k_range:
            kmeans = Kmeans(k=k)
            kmeans.fit(data)
            wcss.append(kmeans.cost)

        plt.figure(figsize=(10, 7))
        plt.plot(k_range, wcss)
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('WCSS')
        plt.xticks(k_range)
        plt.grid(True)
        plt.savefig(f'plots/{plot_name}')
        plt.show()

# def main():
#     Kmeans_header()

if __name__ == "__main__":
    Kmeans_header()  
