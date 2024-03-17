import matplotlib.pyplot as plt
import numpy as np

# Noms des algorithmes de clustering
algorithms = ['K-Means', 'DBSCAN', 'HDBSCAN', 'Agglomération']

# Scores de silhouette pour chaque algorithme
silhouette_scores = [0.355, 0.322, 0.275, 0.322]

# Temps de calcul pour chaque algorithme (en secondes)
calculation_times = [0.148, 0.033, 0.047, 0.033]

# Configuration de la figure et des axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Axe des abscisses pour chaque algorithme, avec un petit décalage pour le deuxième barplot
x = np.arange(len(algorithms))

# Création du barplot pour les scores de silhouette
ax1.bar(x - 0.2, silhouette_scores, width=0.4, label='Score de Silhouette', color='skyblue')
ax1.set_ylabel('Score de Silhouette', color='skyblue')

# Configuration de l'axe secondaire pour les temps de calcul
ax2 = ax1.twinx()
ax2.bar(x + 0.2, calculation_times, width=0.4, label='Temps de Calcul (sec)', color='lightgreen')
ax2.set_ylabel('Temps de Calcul (sec)', color='lightgreen')

# Ajout des titres et des étiquettes
ax1.set_xlabel('Algorithmes de Clustering')
ax1.set_xticks(x)
ax1.set_xticklabels(algorithms)
ax1.set_title('Comparaison des Performances de Clustering')

# Ajout d'une légende
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

plt.show()
