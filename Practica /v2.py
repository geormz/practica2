import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Para gráficos 3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Función para cargar el dataset con perturbaciones en 3D
def dataset_per(filename):
    data = pd.read_csv(filename, header=None)  # Sin encabezados
    return data

# Parámetros
num_particiones = 10
porcebtaje_entre = 0.8

# Ciclo para procesar los tres datasets con perturbaciones
doc_perturbado = ['spheres2d10.csv', 'spheres2d50.csv', 'spheres2d70.csv']

for filename in doc_perturbado:
    # Cargar el dataset con perturbaciones en 3D
    data = dataset_per(filename)
    
    # Listas para almacenar los resultados
    entrenamiento = []
    prueba = []

    # Realizar diez particiones y entrenar modelos
    for _ in range(num_particiones):
        # Dividir el dataset en entrenamiento y prueba
        datos_entre, prueba_datos = train_test_split(data, train_size=porcebtaje_entre)

        # Separar características (X) y etiquetas (y)
        X_train = datos_entre.iloc[:, :-1].values  # Todas las columnas excepto la última
        y_train = datos_entre.iloc[:, -1].values   # La última columna
        X_test = prueba_datos.iloc[:, :-1].values
        y_test = prueba_datos.iloc[:, -1].values

        # Crear el modelo de perceptrón simple
        perceptron = Perceptron(max_iter=1000)

        # Entrenar el modelo
        perceptron.fit(X_train, y_train)

        # Predecir con los datos de entrenamiento y prueba
        y_train_pred = perceptron.predict(X_train)
        y_test_pred = perceptron.predict(X_test)

        # Calcular la precisión en entrenamiento y prueba
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        entrenamiento.append(train_accuracy)
        prueba.append(test_accuracy)

    # Imprimir los resultados para el dataset con perturbaciones actual
    print(f'Resultados para {filename}:')
    print('Precisión en entrenamiento:', np.mean(entrenamiento))
    print('Precisión en prueba:', np.mean(prueba))
    print()
    
    # Visualizar las rectas de distribución de clases en 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c=data.iloc[:, 3], cmap='viridis')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title('Distribución de Clases')
    plt.show()
