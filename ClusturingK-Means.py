import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from sklearn.metrics import silhouette_score
# Importer avec pandas le dataset des drivers
Medical = "Medicaldataset.csv"
data = pd.read_csv(Medical)
print(data.head())

# Appliquer une fonction pour effacer la colonne resultat 

data.drop('Result', axis=1, inplace=True)

# Sélectionner les caractéristiques pour le clustering
features = data[['Age', 'Gender', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']]



# Choisir le nombre de clusters (K)
k = 2

# Mesurer le temps de calcul pour l'exécution de K-means
start_time = time.time()

# Appliquer le clustering K-means
kmeans = KMeans(n_clusters=k)
data['Cluster'] = kmeans.fit_predict(features)
# calcule De la sylhouette 
silhouette_avg = silhouette_score(features, data['Cluster'])
print(f"For k = {k}, the silhouette score is {silhouette_avg}")
# Calculer le temps écoulé
elapsed_time = time.time() - start_time
print(f"Temps de calcul pour K-means: {elapsed_time} secondes")


# Visualiser les clusters
plt.scatter(data['Age'], data['Troponin'], c=data['Cluster'], cmap='viridis', edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 7], s=300, c='red', marker='X', label='Centroide')
plt.xlabel('Age')
plt.ylabel('Heart rate')
plt.title('K-means Clustering of Medical Results')
plt.legend()
plt.show()

    