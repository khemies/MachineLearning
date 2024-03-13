import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import time
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Réimportation des données pour définir X correctement
data = pd.read_csv("Medicaldataset.csv")

# Supprimer les valeurs manquantes si nécessaire
data.dropna(inplace=True)

# Sélectionner les caractéristiques pertinentes, excluant 'Result'
X = data.drop(columns=['Result'])

# Prétraitement des données avec StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Réduction de dimensionnalité avec PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Fonction pour appliquer le clustering hiérarchique agglomératif sur les données réduites par PCA
def hierarchical_clustering_pca(n_clusters, X_pca):
    start_time = time.time()
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(X_pca)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return clustering, elapsed_time

# Visualisation des dendrogrammes sur les données réduites par PCA
plt.figure(figsize=(10, 6))
dendrogram = shc.dendrogram(shc.linkage(X_pca, method='ward'))
plt.title('Dendrogramme avec PCA')
plt.xlabel('Observations')
plt.ylabel('Distance Euclidienne')
plt.show()

# Répétition du processus pour trouver le nombre optimal de clusters avec les données PCA
max_clusters = 10
silhouette_scores_pca = []
computing_times_pca = []

for n_clusters in range(2, max_clusters + 1):
    clustering_pca, elapsed_time = hierarchical_clustering_pca(n_clusters, X_pca)
    silhouette_avg = silhouette_score(X_pca, clustering_pca.labels_)
    silhouette_scores_pca.append(silhouette_avg)
    computing_times_pca.append(elapsed_time)
    print(f"Nombre de clusters : {n_clusters}, Score de silhouette avec PCA : {silhouette_avg}, Temps de calcul : {elapsed_time} secondes")

# Visualisation des scores de silhouette et des temps de calcul pour les données PCA
plt.figure(figsize=(10, 6))
plt.plot(range(2, max_clusters + 1), silhouette_scores_pca, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Score de silhouette avec PCA')
plt.title('Score de silhouette en fonction du nombre de clusters avec PCA')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(2, max_clusters + 1), computing_times_pca, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Temps de calcul (secondes) avec PCA')
plt.title('Temps de calcul en fonction du nombre de clusters avec PCA')
plt.grid(True)
plt.show()
