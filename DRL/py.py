import sys, os
sys.path.append(os.getcwd())
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from DRL.Utilities import delete_files_2
delete_files_2()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

# Step 1: Create or load your dataset (here we create a random dataset)
# X, _ = make_classification(n_samples=1000, n_features=10, random_state=42)

# # Step 2: Perform PCA
# pca = PCA()
# pca.fit(X)

# # Step 3: Get the explained variance (eigenvalues)
# explained_variance = pca.explained_variance_

# cov_matrix = np.cov(X, rowvar=False)
# eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# # Step 4: Plot the scree plot
# plt.figure(figsize=(8, 6))
# plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
# plt.title('Scree Plot')
# plt.xlabel('Principal Components')
# plt.ylabel('Eigenvalues / Explained Variance')
# plt.grid(True)
# plt.show()




# Generate some synthetic data
# np.random.seed(42)
# data = np.random.rand(1000, 160)

# # Calculate WCSS for a range of number of clusters
# wcss = []
# for i in range(5, 1000, 5):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(data)
#     wcss.append(kmeans.inertia_)

# # Create the scree plot
# plt.figure(figsize=(8, 5))
# plt.plot(range(5, 1000, 5), wcss, 'bo-')
# plt.title('Elbow Method For Optimal k')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
# plt.grid(True)
# plt.show()





# from sklearn.decomposition import PCA
# from sklearn.datasets import load_iris

# # Load example data
# data = load_iris()
# #X = data.data
# X = np.random.rand(100,250)

# # Perform PCA
# pca = PCA()
# X_pca = pca.fit_transform(X)


# # Eigenvalues (variance explained by each component)
# explained_variance = pca.explained_variance_

# # Creating the scree plot
# plt.figure(figsize=(8, 5))
# plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-')
# plt.title('Scree Plot')
# plt.xlabel('Principal Component')
# plt.ylabel('Eigenvalue (Variance Explained)')
# plt.xticks(range(1, len(explained_variance) + 1))
# plt.grid(True)
# plt.show()























# Generate some random data

# np.random.seed(4)
# data_high_D = np.random.rand(1000,160) * 50  # 100 points in 2D space
# tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=42)
# data = tsne.fit_transform(data_high_D)
# #data = data_high_D
# #print(data)
# # Create the KMedoids model
# k = 45  # number of clusters
# kmeans = KMeans(n_clusters=k, random_state=0, max_iter=1000)
# kmeans.fit(data)
# initial_clusters = kmeans.predict(data)
# centroids = []
# centroids_indices = []
# clusters_all = []
# for i in range(k):
#     clustered_data =  data[initial_clusters == i]
#     clusters_all.append(clustered_data)
#     kmedoids = KMedoids(n_clusters=1, random_state=0, max_iter = 1000)
#     kmedoids.fit(clustered_data)
#     medoid = kmedoids.cluster_centers_
#     matches = np.all(data == medoid, axis=1)
#     index = np.where(matches)[0]
#     centroids.append(medoid)
#     centroids_indices.append(index)

# centroids_indices_np = np.array(centroids_indices).reshape(k,-1)
# centroids_np = np.array(centroids).reshape(k,-1)

# print("Medoids (indices):", centroids_indices_np)
# print("Medoids (coordinates):\n", centroids_np)

# # Plotting
# plt.scatter(data[:, 0], data[:, 1], c=initial_clusters, cmap='viridis', marker='o')
# plt.scatter(centroids_np[:, 0], centroids_np[:, 1], c='red', s=100, marker='x')  # mark the medoids
# plt.title('K-Means Clustering with K-Medoids Centroids')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.show()

