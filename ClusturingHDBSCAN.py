import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.metrics import silhouette_score
import time

# Charger les données à partir du fichier CSV
data = pd.read_csv("Medicaldataset.csv")

# Supprimer les valeurs manquantes si nécessaire
data.dropna(inplace=True)

# Sélectionner les caractéristiques pertinentes
X = data.drop(columns=['Result'])

# Appliquer HDBSCAN avec des valeurs de paramètres (min_samples, eps) choisies au hasard
min_samples = 10
eps = 2
start_time = time.time()
clusterer = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=eps)
clustering = clusterer.fit(X)
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Temps d'execution : {elapsed_time}")
# Visualiser les résultats du clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=clustering.labels_, palette='viridis')
plt.xlabel('Age')
plt.ylabel('Heart rate')
plt.title('Résultats du clustering HDBSCAN')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

# Utiliser les métriques d'évaluation pour analyser la qualité du clustering
silhouette_avg = silhouette_score(X, clustering.labels_)
print(f"Score de silhouette : {silhouette_avg}")
