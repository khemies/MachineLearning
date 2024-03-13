import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
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

# Appliquer HDBSCAN avec des paramètres ajustés
min_samples = 10
min_cluster_size = 2  # Ajuster en fonction des données
start_time = time.time()
clusterer = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
clustering = clusterer.fit(X_pca)
end_time = time.time()

# Temps d'exécution
elapsed_time = end_time - start_time
print(f"Temps d'exécution : {elapsed_time} secondes")

# Visualiser les résultats du clustering
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clustering.labels_, palette='viridis')
plt.xlabel('Première composante principale')
plt.ylabel('Deuxième composante principale')
plt.title('Résultats du clustering HDBSCAN')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

# Calcul du score de silhouette
if len(np.unique(clustering.labels_)) > 1:  # Éviter une erreur si un seul cluster est trouvé
    silhouette_avg = silhouette_score(X_pca, clustering.labels_)
    print(f"Score de silhouette : {silhouette_avg}")
else:
    print("Le calcul du score de silhouette n'est pas applicable pour un seul cluster.")
