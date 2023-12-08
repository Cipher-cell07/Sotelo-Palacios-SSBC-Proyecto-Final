import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Cargar dataset
data = pd.read_csv('Employee.csv')

# Seleccionar características (incluyendo variables categóricas)
X = data[['JoiningYear', 'PaymentTier', 'Age', 'EverBenched', 'ExperienceInCurrentDomain', 'Education', 'City', 'Gender']]
y = data['LeaveOrNot']

# Convertir variables categóricas utilizando one-hot encoding
X = pd.get_dummies(X)

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos después de dividir
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# ----------------------- k-NN -----------------------
def hamming_distance(x1, x2):
    return np.sum(x1 != x2)

def k_nearest_neighbors(X_train, y_train, X_test, k=3):
    predictions = []
    for x in X_test.values:
        distances = [hamming_distance(x, x_train) for x_train in X_train.values]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train.iloc[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        predictions.append(most_common)
    return predictions

# Entrenar el modelo k-NN
k = 3
y_pred = k_nearest_neighbors(X_train, y_train, X_test, k)

# Calcular la precisión para el conjunto de prueba
accuracy_test = np.sum(y_pred == y_test) / len(y_test)
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