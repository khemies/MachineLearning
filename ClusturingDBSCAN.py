import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import time

# Charger les données à partir du fichier CSV
data = pd.read_csv("Medicaldataset.csv")

# Supprimer les valeurs manquantes si nécessaire
data.dropna(inplace=True)

# Sélectionner les caractéristiques pertinentes
X = data.drop(columns=['Result'])

# Appliquer DBSCAN avec des valeurs de paramètres (min_samples, eps) choisies au hasard
min_samples =10
eps = 12
start_time = time.time()
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
clustering = dbscan.fit(X)
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)
# Visualiser les résultats du clustering
labels = clustering.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Nombre de clusters : {n_clusters}")

# Visualisation
plt.figure(figsize=(10, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Heart rate')
plt.title('Résultats du clustering DBSCAN')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Vérifier les étiquettes de cluster
print(labels)

# Assurez-vous qu'il n'y a pas de valeurs négatives dans les étiquettes
if np.any(labels < 0):
    print("Erreur : Les étiquettes de cluster contiennent des valeurs négatives.")
else:
    # Calculer le score de silhouette
    silhouette_avg = silhouette_score(X, labels)
    print(f"Score de silhouette : {silhouette_avg}")



