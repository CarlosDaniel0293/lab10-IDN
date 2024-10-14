import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Cargar el archivo Excel
data = pd.read_excel('Data10-1.xlsx')

# Visualizar los primeros datos del dataframe
print(data.head())

# Preprocesar los datos
# Convertir 'Estado' a variables numéricas
data['Estado'] = data['Estado'].map({'Alto': 1, 'Bajo': 0})

# Seleccionar características y etiquetas
X = data[['Precio actual', 'Precio final']].values
y = data['Estado'].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el clasificador SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Hacer predicciones
y_pred = svm_model.predict(X_test)

# Evaluar el modelo
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Visualización de los resultados
def visualize_svm(X, y, model):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', edgecolors='k')

    # Crear una cuadrícula
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Predecir el resultado para la cuadrícula
    Z = model.predict(scaler.transform(grid)).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2)

    plt.title("SVM Classifier Decision Boundary")
    plt.xlabel("Precio Actual")
    plt.ylabel("Precio Final")
    plt.show()

# Visualizar el SVM
visualize_svm(X_train, y_train, svm_model)
