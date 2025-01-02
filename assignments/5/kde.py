# from sklearn.mixture import GaussianMixture
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))  
from models.GMM.GMM import GMM
from models.KDE.KDE import KDE

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal

class kde_file:
    def main():
        X = np.array([[1, 2], [2, 3], [5, 6], [6, 7], [10, 15], [8, 9], [9, 10], [10, 11]])

        kde = KDE(kernel_type='gaussian', bandwidth=2)
        kde.fit(X)

        density = kde.predict(X)

        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=density, cmap='viridis', s=100 + density * 200, edgecolor='k')
        plt.colorbar(label='Density')
        plt.title('2D KDE Estimation with Gaussian Kernel', fontsize=16)
        plt.xlabel('X-axis', fontsize=14)
        plt.ylabel('Y-axis', fontsize=14)
        plt.show()

        def generate_random_in_circle(radius = 1,offset = 0):
            r = radius * np.sqrt(np.random.rand())
            theta = np.random.rand() * 2 * np.pi
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            return x+offset, y+offset

        radius = 2
        big_circle = np.array([generate_random_in_circle(radius) for _ in range(3000)])
        sigma1 = radius * 0.1
        big_circle = big_circle + sigma1*np.random.randn(*big_circle.shape)

        small_radius = 0.25
        small_circle = np.array([generate_random_in_circle(radius=small_radius,offset=1) for _ in range(300)])
        sigma2 = small_radius * 0.1
        small_circle = small_circle + sigma2*np.random.randn(*small_circle.shape)

        X = np.array(list(big_circle) + list(small_circle))

        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], s=2, edgecolor='k')
        plt.title('Random Points in Circle', fontsize=16)
        plt.xlabel('X-axis', fontsize=14)   
        plt.ylabel('Y-axis', fontsize=14)
        plt.axis('equal')
        plt.grid(True)
        plt.show()

        kde = KDE(kernel_type='gaussian', bandwidth=0.4)
        kde.fit(X)

        density = kde.predict(X)

        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=density, cmap='viridis', s=8, edgecolor='k')
        plt.colorbar(label='Density')
        plt.title('2D KDE Estimation with Gaussian Kernel', fontsize=16)
        plt.xlabel('X-axis', fontsize=14)
        plt.ylabel('Y-axis', fontsize=14)
        plt.show()

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))

        grid_density = kde.predict(np.c_[xx.ravel(), yy.ravel()])
        grid_density = grid_density.reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=density, cmap='viridis', s=8, edgecolor='k')
        plt.contourf(xx, yy, grid_density, 20, cmap='viridis', alpha=0.5)
        plt.colorbar(label='Density')
        plt.title('2D KDE Estimation with Gaussian Kernel', fontsize=16)
        plt.xlabel('X-axis', fontsize=14)
        plt.ylabel('Y-axis', fontsize=14)
        plt.show()

        gmm = GMM(k=2)
        gmm.fit(X)

        cluster = gmm.predict(X)
        print(cluster)

        prob = gmm.predict_proba(X)

        levels = np.digitize(prob, bins=np.linspace(0, 1, 10)) - 1

        print(levels[0])

        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=cluster, s=10, cmap="viridis", edgecolor='k')
        plt.title('GMM Clustering Results with 2 Components', fontsize=16)
        plt.xlabel('X-axis', fontsize=14)
        plt.ylabel('Y-axis', fontsize=14)
        plt.axis('equal')
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=levels[:,0], s=10, cmap="viridis", edgecolor='k')
        plt.title('GMM Clustering Results with 2 Components with predict_proba', fontsize=16)
        plt.xlabel('X-axis', fontsize=14)
        plt.ylabel('Y-axis', fontsize=14)
        plt.axis('equal')
        plt.show()


        sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
        from models.GMM.GMM import GMM

        # Create 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        fig2,axes2 = plt.subplots(2, 2, figsize=(16, 12))

        for n in np.arange(2, 6):
            gmm = GMM(k=n)
            gmm.fit(X)

            cluster = gmm.predict(X)
            print(cluster)
            
            prob = gmm.predict_proba(X)

            levels = np.digitize(prob, bins=np.linspace(0, 1, 10)) - 1

            print(levels[0])
            
            ax = axes[(n-2)//2, (n-2)%2]  
            ax.scatter(X[:, 0], X[:, 1], c=cluster, s=10, cmap="viridis", edgecolor='k')
            ax.set_title(f'GMM Clustering Results with {n} Components', fontsize=16)
            ax.set_xlabel('X-axis', fontsize=14)
            ax.set_ylabel('Y-axis', fontsize=14)
            ax.axis('equal')
            
            ax2 = axes2[(n-2)//2, (n-2)%2] 
            ax2.scatter(X[:, 0], X[:, 1], c=levels[:,0], s=10, cmap="viridis", edgecolor='k')
            ax2.set_title(f'GMM Clustering Results with {n} Components with predict_proba', fontsize=16)
            ax2.set_xlabel('X-axis', fontsize=14)
            ax2.set_ylabel('Y-axis', fontsize=14)
            ax2.axis('equal')

        plt.show()
