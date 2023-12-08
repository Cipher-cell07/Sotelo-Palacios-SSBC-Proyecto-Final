import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Cargar dataset
data = pd.read_csv('zoo.csv')

# Seleccionar solo las características numéricas
columns = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize', 'class_type']
selected_data = data[columns]

# Verificar los tipos de datos y manejar conversiones, verificar datos faltantes
print(selected_data.info())

# Dividir los datos en características (X) y la variable a predecir (y)
X = selected_data.drop('class_type', axis=1)
y = selected_data['class_type']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos después de dividir
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# ----------------------- k-NN -----------------------
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def k_nearest_neighbors(X_train, y_train, X_test, k=3):
    predictions = []
    for x in X_test.values:
        distances = [euclidean_distance(x, x_train) for x_train in X_train.values]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train.iloc[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        predictions.append(most_common)
    return predictions

k = 3
# Realizar predicciones para todo el conjunto de datos de prueba
y_pred = k_nearest_neighbors(X_train, y_train, X_test, k)

# Calcular la precisión para el conjunto de prueba
accuracy_test = accuracy_score(y_test, y_pred)
print(f'Precisión para k-NN en el conjunto de prueba con k={k}: {accuracy_test}')

# Calcular el informe de clasificación que incluye precisión, recall, etc
informe = classification_report(y_test, y_pred, zero_division=1)

# Imprimir el informe de clasificación
print("\nInforme:")
print(informe)

#Graficar

result_df = pd.DataFrame({'Real': y_test.values, 'Predicho': y_pred})

plt.scatter(result_df.index, result_df['Real'], label='Real', marker='o')
plt.scatter(result_df.index, result_df['Predicho'], label='Predicho', marker='x')
plt.xlabel('Índice de la muestra')
plt.ylabel('Clase')
plt.title('Valores Reales vs. Predichos')
plt.legend()
plt.show()