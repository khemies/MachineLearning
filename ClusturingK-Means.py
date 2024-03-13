import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
from sklearn.metrics import silhouette_score

# Charger le jeu de données
Medical = "Medicaldataset.csv"  # Assurez-vous de mettre à jour le chemin
data = pd.read_csv(Medical)
print(data.head())

# Supprimer la colonne 'Result'
data.drop('Result', axis=1, inplace=True)

# Sélectionner les caractéristiques pour le clustering
features = data[['Age', 'Gender', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']]

# Standardiser les caractéristiques
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Appliquer PCA pour réduire la dimensionnalité à 2 composantes
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# Choisir le nombre de clusters (K)
k = 2

# Mesurer le temps de calcul pour l'exécution de K-means
start_time = time.time()

# Appliquer le clustering K-means sur les données réduites par PCA
kmeans = KMeans(n_clusters=k, random_state=42)
data['Cluster'] = kmeans.fit_predict(features_pca)

# Calculer la silhouette score
silhouette_avg = silhouette_score(features_pca, data['Cluster'])
print(f"For k = {k}, the silhouette score is {silhouette_avg}")

# Calculer le temps écoulé
elapsed_time = time.time() - start_time
print(f"Temps de calcul pour K-means: {elapsed_time} secondes")

# Visualiser les clusters sur les deux premières composantes principales
plt.figure(figsize=(10, 6))
plt.scatter(features_pca[:, 0], features_pca[:, 1], c=data['Cluster'], cmap='viridis', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', label='Centroids')

plt.xlabel('Première composante principale')
plt.ylabel('Deuxième composante principale')
plt.title('Clusters visualisés selon l algorithme K-means avec PCA')
plt.legend()
plt.show()
