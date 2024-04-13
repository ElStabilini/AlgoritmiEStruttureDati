from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

## Create a graph using networkx
#G = nx.Graph()
#G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)])
#
## Convert networkx graph to adjacency matrix
#adj_matrix = nx.adjacency_matrix(G).toarray()
#
## Perform spectral clustering for partitioning
#sc = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
#labels = sc.fit_predict(adj_matrix)
#
## Display partitioning result
#print("Partitioning result:", labels)

# get a set of data points
df = pd.read_csv('./datasets/Aggregation.csv', sep=' ',
                     header=None)

df.columns = ['x', 'y']

# Create feature matrix
X = df.values

# Perform KMeans clustering
kmeans = KMeans(n_clusters=7, random_state=42)
labels = kmeans.fit_predict(X)

# Display clustering result
print("Clustering result:", labels)

# Plot the clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='x', y='y', hue=labels, palette='rainbow', edgecolor='k', s=50)
plt.title('KMeans Clustering')
plt.show()