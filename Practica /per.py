import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Cargar el dataset
datos = pd.read_csv('spheres1d10.csv')

# Parámetros
num_particiones = 2
porcentaje = 0.1

# Listas para almacenar los resultados
entrenamiento = []
prueba = []

# Realizar cinco particiones y entrenar modelos
for _ in range(num_particiones):
    # Dividir el dataset en entrenamiento y prueba
    datos_entre, prueba_datos = train_test_split(datos, train_size=porcentaje)

    # Separar características (X) y etiquetas (y)
    X_train = datos_entre.iloc[:, :-1].values
    y_train = datos_entre.iloc[:, -1].values
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

# Graficar los resultados
plt.figure(figsize=(8, 4))
plt.bar(range(1, num_particiones + 1), entrenamiento, label='Entrenamiento', width=0.4)
plt.bar(np.arange(1, num_particiones + 1) + 0.4, prueba, label='Prueba', width=0.4)
plt.xlabel('Partición')
plt.ylabel('Precisión')
plt.title('Precisión del Perceptrón Simple en Entrenamiento y Prueba')
plt.xticks(np.arange(1, num_particiones + 1))
plt.legend(loc='best')
plt.show()
