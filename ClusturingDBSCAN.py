import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time

# Charger les données à partir du fichier CSV
data = pd.read_csv("Medicaldataset.csv")

# Supprimer les valeurs manquantes si nécessaire
data.dropna(inplace=True)

# Sélectionner les caractéristiques pertinentes et retirer la colonne 'Result'
X = data.drop(columns=['Result'])

# Standardisation des caractéristiques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Réduction de dimensionnalité avec PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Appliquer DBSCAN avec des valeurs de paramètres (min_samples, eps) ajustées
min_samples = 10  # À ajuster selon vos données
eps = 12  # À ajuster selon vos données
start_time = time.time()
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
clustering = dbscan.fit(X_pca)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Temps d'exécution : {elapsed_time} secondes")

# Visualiser les résultats du clustering
labels = clustering.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Nombre de clusters : {n_clusters}")

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.xlabel('Première composante principale')
plt.ylabel('Deuxième composante principale')
plt.title('Résultats du clustering DBSCAN')
plt.colorbar(label='Cluster ID')
plt.grid(True)
plt.show()

# Calculer le score de silhouette uniquement si applicable
if n_clusters > 1:
    silhouette_avg = silhouette_score(X_pca, labels)
    print(f"Score de silhouette : {silhouette_avg}")
else:
    print("Le calcul du score de silhouette n'est pas applicable.")
