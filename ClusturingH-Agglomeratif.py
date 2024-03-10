import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import time
import scipy.cluster.hierarchy as shc

# Étape 1 : Charger les données à partir du fichier CSV
data = pd.read_csv("Medicaldataset.csv")

# Supprimer les valeurs manquantes si nécessaire
data.dropna(inplace=True)

# Sélectionner les caractéristiques pertinentes
X = data.drop(columns=['Result'])


# Étape 2 : Appliquer le clustering hiérarchique agglomératif avec un nombre prédéfini de clusters
def hierarchical_clustering(n_clusters):
    start_time = time.time()
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return clustering, elapsed_time

# Étape 3 : Visualiser les dendrogrammes
plt.figure(figsize=(10, 6))
dendrogram = shc.dendrogram(shc.linkage(X, method='ward'))
plt.title('Dendrogramme')
plt.xlabel('Observations')
plt.ylabel('Distance Euclidienne')
plt.show()

# Étape 4 : Utiliser les métriques d'évaluation pour analyser la qualité du clustering
def evaluate_clustering(clustering):
    silhouette_avg = silhouette_score(X, clustering.labels_)
    return silhouette_avg

# Étape 5 : Itérer pour trouver le nombre optimal de clusters et mesurer le temps de calcul
max_clusters = 10
silhouette_scores = []
computing_times = []

for n_clusters in range(2, max_clusters + 1):
    clustering, elapsed_time = hierarchical_clustering(n_clusters)
    silhouette_avg = evaluate_clustering(clustering)
    silhouette_scores.append(silhouette_avg)
    computing_times.append(elapsed_time)
    print(f"Nombre de clusters : {n_clusters}, Score de silhouette : {silhouette_avg}, Temps de calcul : {elapsed_time} secondes")

# Visualiser les résultats
plt.figure(figsize=(10, 6))
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Score de silhouette')
plt.title('Score de silhouette en fonction du nombre de clusters')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(2, max_clusters + 1), computing_times, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Temps de calcul (secondes)')
plt.title('Temps de calcul en fonction du nombre de clusters')
plt.grid(True)
plt.show()

