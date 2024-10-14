import numpy as np
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Inicializamos los parámetros
        self.w = np.zeros(n_features)
        self.b = 0

        # Convertimos las etiquetas a 1 y -1
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Condición de margen
                if y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1:
                    # Actualización de parámetros
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # Actualización de parámetros
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

# Generación de datos de ejemplo
def create_dataset(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = np.array([1 if x[0] + x[1] > 0 else -1 for x in X])
    return X, y

# Visualización de los datos
def visualize_svm(X, y, svm):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')

    # Limites del gráfico
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Crear una cuadrícula
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = svm.predict(grid).reshape(xx.shape)

    # Contornos
    ax.contourf(xx, yy, preds, alpha=0.2)

    plt.title("SVM Classifier")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Ejecución del SVM
if __name__ == "__main__":
    X, y = create_dataset(100)
    svm = SVM()
    svm.fit(X, y)
    visualize_svm(X, y, svm)
