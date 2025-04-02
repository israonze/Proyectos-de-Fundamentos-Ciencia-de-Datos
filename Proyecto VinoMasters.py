#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importrar librerías
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar los datos
df = pd.read_csv('datos_VinoMaster.csv')

#Imprimir las primeras filas del archivo y una descripción básica
print(df.head())
print(df.describe())
print("Nombres de las columnas:", df.columns.tolist())
print("Valores nulos por columna:\n", df.isnull().sum())

#Realizar un histograma con la distribución de la variable quality
plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=df, palette='viridis')
plt.title('Distribución de la Calidad del Vino')
plt.show()

# Obtener solo las columnas numéricas
numeric_cols = df.select_dtypes(include=['number']).columns

# Separar en grupos de 3
num_cols = len(numeric_cols)
num_plots = int(np.ceil(num_cols / 3))  # Número total de gráficos


# Crear gráficos en grupos de 3 variables
for i in range(num_plots):
    plt.figure(figsize=(12, 6))
    cols = numeric_cols[i*3 : (i+1)*3]  # Selecciona 3 columnas por iteración
    sns.boxplot(data=df[cols])
    
    plt.title(f"Boxplot de Variables ({i*3+1} a {(i+1)*3})")
    plt.xticks(rotation=45)
    plt.show()

#Usa la función drop para quitar los atributos necesarios.
X = df.drop(['id', 'quality'], axis=1)
y = df['quality']

#Convierte la variable categórica Type en variable numérica.
X['Type'] = (X['Type'] == 'red').astype(int)

#Separa los datos en conjuntos de entrenamiento y conjunto de pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

#Realiza un escalamiento de los datos para asegurarte que todas las variables numéricas estén en los mismos rangos de valores.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Define el modelo usando
nn_model = MLPRegressor(hidden_layer_sizes=(6,6,6), max_iter=500, random_state=42)

#Ajusta el modelo a los datos de entrenamiento usando la función
nn_model.fit(X_train_scaled, y_train)

#Calcula las predicciones del modelo usando
nn_pred = nn_model.predict(X_test_scaled)

#Evalúa el modelo usando error cuadrado medio, MSE y R-cuadrada
nn_mse = mean_squared_error(y_test, nn_pred)
nn_r2 = r2_score(y_test, nn_pred)
print("Resultados de la Red Neuronal:")
print(f"Error Cuadrático Medio: {nn_mse}")
print(f"R-cuadrado: {nn_r2}")

# Crear y entrenar el modelo de regresión lineal
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Hacer predicciones con la regresión lineal
lr_pred = lr_model.predict(X_test_scaled)

# Comparar las predicciones con los valores reales
resultados = pd.DataFrame({'Real': y_test.values, 'Predicción': lr_pred})
print(resultados.head())  # Mostrar las primeras filas

# Evaluar la regresión lineal
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
print("\nResultados de la Regresión Lineal:")
print(f"Error Cuadrático Medio: {lr_mse}")
print(f"R-cuadrado: {lr_r2}")


# In[ ]:




