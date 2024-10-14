import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Cargar el archivo Excel
data = pd.read_excel('Data10-1.xlsx')

# Visualizar los primeros datos del dataframe
print(data.head())

# Preprocesar los datos
# Codificar la columna 'Estado' como variable objetivo
label_encoder = LabelEncoder()
data['Estado'] = label_encoder.fit_transform(data['Estado'])

# Seleccionar características para el modelo
X = data[['Precio actual', 'Precio final']]
y = data['Estado']  # variable objetivo

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Aplicar el árbol de decisión
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predecir con el conjunto de prueba
y_pred = clf.predict(X_test)

# Evaluar el modelo
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Visualizar el árbol de decisión
from sklearn import tree

plt.figure(figsize=(12,8))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=label_encoder.classes_)
plt.title('Árbol de Decisión')
plt.show()
