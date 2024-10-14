import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cargar el archivo Excel
data = pd.read_excel('Data10-1.xlsx')

# Visualizar los primeros datos del dataframe
print(data.head())

# Preprocesar los datos
# Seleccionar características para la agrupación
X = data[['Precio actual', 'Precio final']].values

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=42)  # Puedes ajustar el número de clústeres
kmeans.fit(X_scaled)

# Obtener las etiquetas de los clústeres
labels = kmeans.labels_

# Agregar las etiquetas al dataframe original
data['Cluster'] = labels

# Visualizar los resultados
def plot_kmeans(X, labels, centers):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolors='k')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroides')
    plt.title('K-Means Clustering')
    plt.xlabel('Precio Actual')
    plt.ylabel('Precio Final')
    plt.legend()
    plt.show()

# Obtener los centroides
centers = kmeans.cluster_centers_

# Visualizar los resultados
plot_kmeans(X_scaled, labels, centers)

# Mostrar el dataframe con los clústeres
print(data[['Precio actual', 'Precio final', 'Cluster']].head())
