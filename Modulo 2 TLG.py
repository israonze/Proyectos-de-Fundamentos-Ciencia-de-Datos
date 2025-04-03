#!/usr/bin/env python
# coding: utf-8

# In[3]:


from numpy import array
from numpy.linalg import det
A = array([[-3,2,1,-4],
 [1,3,0,-3],
 [-3,4,-2,8],
 [3,-4,0,4]])
determinante = det(A)
print('A = \n', A)
print('det(A)=',determinante)


# In[2]:


import numpy as np
from numpy.linalg import inv
A = np.array([[2, 5], [-3, -7]])
B = inv(A)
5
print(' A ='); print(A)
print(' B ='); print(B)
print('AB ='); print(np.dot(A,B))


# In[4]:


import numpy as np
from numpy.linalg import inv
A = np.array([[0.86, 0, 0.78], [0.01,
0.92, 0.84], [0.09, 0.63, 4.5]])
b = np.array([5000, 2000,3000])
B = inv(A)
v = np.dot(B,b)
print (v)


# In[5]:


import numpy as np

# Definir las matrices
E1 = np.array([[0, 0, 1, 0],
               [1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1]])

E2 = np.array([[1, 0, 0, 0],
               [0, 0, 1, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1]])

E3 = np.array([[1, 0, 0, 0],
               [0, 0, 1, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 2]])

# Calcular determinantes
det_E1 = np.linalg.det(E1)
det_E2 = np.linalg.det(E2)
det_E3 = np.linalg.det(E3)

# Calcular inversas si el determinante no es cero
inv_E1 = np.linalg.inv(E1) if det_E1 != 0 else "No tiene inversa"
inv_E2 = np.linalg.inv(E2) if det_E2 != 0 else "No tiene inversa"
inv_E3 = np.linalg.inv(E3) if det_E3 != 0 else "No tiene inversa"

# Imprimir resultados
print("Determinantes:")
print(f"det(E1) = {det_E1}")
print(f"det(E2) = {det_E2}")
print(f"det(E3) = {det_E3}")

print("\nInversas:")
print("Inversa de E1:")
print(inv_E1)

print("\nInversa de E2:")
print(inv_E2)

print("\nInversa de E3:")
print(inv_E3)


# In[9]:


from sklearn import datasets
import matplotlib.pyplot as plt

# Cargar el dataset de iris
iris = datasets.load_iris()

# Crear la figura y los ejes
fig, ax = plt.subplots()

# Graficar los datos
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)

# Etiquetas de los ejes
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])

# Agregar leyenda
legend = ax.legend(scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Especies")

# Mostrar el gráfico
plt.show()


# In[1]:


import matplotlib.pyplot as plt
import numpy as np
def ReLU(x):
 return np.maximum(0, x)
x = np.linspace(-5, 5, 1000)
plt.figure(figsize=(8, 4))
plt.plot(x, ReLU(x), label='ReLU', color='blue')
plt.title('Función de activación')
plt.grid(True)
plt.legend()
plt.show()


# # RED NEURONAL EJ

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPClassifier

df= pd.read_csv('datosRGB001.csv', delimiter= ",")


#extraer las variables de entrada (todas menos la última)
#Hacer un escalamiento para que X esté entre 0 y 1.
X = (df.values[:, :-1] / 255.0) 

#Extraer la últma colmna 
Y= df.values [:,-1]

#Separar el conjnto en tranining y testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/4)

nn = MLPClassifier(solver='sgd',
                   hidden_layer_sizes=(3, 2),
                   activation='relu',
                   max_iter=100000,
                   learning_rate_init=0.01) 

#Entrenar la red neuronal
nn.fit(X_train, Y_train)

loss_curve = nn.loss_curve_

plt.figure(figsize=(5, 3))
plt.plot(loss_curve, label='Función de Pérdida')
plt.title('Función de Pérdida vs Iteraciones')
plt.xlabel('Iteraciones')
plt.ylabel('Valor de la Función de Pérdida')
plt.legend()
plt.grid(True)
plt.show() 


# In[10]:


print("Pesos de las capas:", nn.coefs_)
print("Sesgos de las capas:", nn.intercepts_)
print("Número de capas:", nn.n_layers_)
print("Número de iteraciones realizadas:", nn.n_iter_)
print("Función de pérdida:", nn.loss_)
print("Evaluación (conjunto entrenamiento): %f"
% nn.score(X_train, Y_train))
print("Evaluación (conjunto de pruebas): %f" %
nn.score(X_test, Y_test)) 


# In[11]:


nuevos_datos = {
 'R': [63, 69, 76],
 'G': [170, 178, 188],
 'B': [250, 251, 251]
}
# Convertir los nuevos datos a un DataFrame
df_nuevos = pd.DataFrame(nuevos_datos)
# Normalizar los nuevos datos
X_nuevos = df_nuevos.values / 255.0
# Hacer predicciones con el modelo entrenado
predicciones = nn.predict(X_nuevos)
# Mostrar las predicciones
print("Predicciones para los nuevos datos:",
predicciones)


# # Prueba

# In[12]:


from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


# In[13]:


nn= MLPClassifier(solver='sgd',
                 hidden_layer_sizes=(3,2),
                 activation='relu',
                 max_iter=100000,
                 learning_rate_init=0.01)
lr= LogisticRegression(max_iter=100000)


# In[14]:


nn.fit(X_train, Y_train)
print("Evaluation: %f"%
     nn.score(X_test,Y_test))
lr.fit(X_train, Y_train)
print("Evaluation: %f"%
     lr.score(X_test,Y_test))


# In[16]:


##Examen

##Con base en el siguiente código, ¿por qué no es posible llamar a la función fit antes de la función MLPClassifier 

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split 

from sklearn.neural_network import MLPClassifier 

 

df = pd.read_csv('datosRGB001.csv', delimiter=",") 

X = (df.values[:, :-1] / 255.0) 

Y = df.values[:, -1] 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5) 

 

nn = MLPClassifier(solver='sgd', 

                   hidden_layer_sizes=(3, 2), 

                   activation='relu', 

                   max_iter=100000, 

                   learning_rate_init=0.01) 

 

nn.fit(X_train, Y_train) 


# In[17]:


import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split 

from sklearn.neural_network import MLPClassifier 

 

df = pd.read_csv('datosRGB001.csv', delimiter=",") 

X = (df.values[:, :-1] / 255.0) 

Y = df.values[:, -1] 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5) 

 

nn = MLPClassifier(solver='sgd', 

                   hidden_layer_sizes=(3, 2), 

                   activation='relu', 

                   max_iter=100000, 

                   learning_rate_init=0.01) 

 

nn.fit(X_train, Y_train) 


# # Ejercicio práctico 1

# In[2]:


#Esta línea importa librerías y funciones necesarias de Pandas y Scikit-Learn.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:


#Esta línea carga el archivo de datos en formato CSV.
data = pd.read_csv(‘datos_VinoMaster.csv’)


# In[ ]:


#En esta línea se excluyen los atributos id y quality de las variables de entrada al modelo. Se define quality como la variable
#de salida del modelo y se transforma la variable Type en numérica asignando 1 al vino rojo y 0 al vino blanco
X = data.drop([‘id’, ‘quality’], axis=1)
y = data[‘quality’]
X[‘Type’] = (X[‘Type’] == ‘red’).astype(int)


# In[ ]:


#A. Esta línea separa los datos en un conjunto de entrenamiento (80%) y un conjunto de pruebas (20%).7
X_train, X_test, y_train, y_test = train_
test_split(X, y, test_size=0.2, random_
state=42)


# In[ ]:


#D. StandardScaler permite hacer un escalamiento de nuestros datos haciendo que todos tengan media cero y varianza 1. 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[ ]:


#H. En esta línea se crea una red neuronal con una sola capa con 10 neuronas en ella. El MLPRegressor se usa porque la variable
#a predecir es numérica y fit ajusta los parámetros del modelo.
model = MLPRegressor(hidden_layer_
sizes=(10,), max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)


# In[ ]:


#C. predict se usa para calcular las predicciones y las guarda en el vector y_pred.
y_pred = model.predict(X_test_scaled)


# In[ ]:


#F. Estas dos líneas sirven para calcular dos métricas para evaluar nuestro modelo. MSE calcula el error cuadrado medio y R2 es
#un valor estadístico que corresponde a la proporción de la varianza en Y que estaría asociada a las variables x.
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# # Ejercicio practico 2

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# 1. Cargar los datos
data = pd.read_csv(‘datos_VinoMaster.csv’)
# 2. Preparar las características y el objetivo
X = data.drop([‘id’, ‘quality’], axis=1)
y = data[‘quality’]
# 3. Convertir ‘Type’ categórico a numérico
X[‘Type’] = (X[‘Type’] == ‘red’).astype(int)
# 4. Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_
size=0.2, random_state=42)
# 5. Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 6. Crear y entrenar la red neuronal
nn_model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000,
random_state=42)
nn_model.fit(X_train_scaled, y_train)
# 7. Hacer predicciones con la red neuronal
nn_pred = nn_model.predict(X_test_scaled)
# 8. Evaluar la red neuronal
nn_mse = mean_squared_error(y_test, nn_pred)
nn_r2 = r2_score(y_test, nn_pred)
print(“Resultados de la Red Neuronal:”)
print(f”Error Cuadrático Medio: {nn_mse}”)
print(f”R-cuadrado: {nn_r2}”)
# 9. Crear y entrenar el modelo de regresión lineal
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
# 10. Hacer predicciones con la regresión lineal
lr_pred = lr_model.predict(X_test_scaled)
# 11. Evaluar la regresión lineal
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
print(“\n Resultados de la Regresión Lineal:”)
print(f”Error Cuadrático Medio: {lr_mse}”)
print(f”R-cuadrado: {lr_r2}”)
# 12. Comparar los resultados
print(“\n Comparación:”)
print(f”Mejora en MSE: {(lr_mse - nn_mse) / lr_mse * 100:.2f}%”)
print(f”Mejora en R-cuadrado: {(nn_r2 - lr_r2) / lr_r2 *
100:.2f}%”)


# # Reto | Definiendo un modelo predictivo para la evaluación de la calidad del vino

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[3]:


# 1. Cargar los datos
df = pd.read_csv('datos_VinoMaster.csv')


# In[5]:


#Imprimir las primeras filas del archivo y una descripción básica
print(df.head())
print(df.describe())


# In[6]:


print("Nombres de las columnas:", df.columns.tolist())
print("Valores nulos por columna:\n", df.isnull().sum())


# In[31]:


#Realizar un histograma con la distribución de la variable quality
plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=df, palette='viridis')
plt.title('Distribución de la Calidad del Vino')
plt.show()


# In[74]:


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


# In[266]:


#Usa la función drop para quitar los atributos necesarios.
X = df.drop(['id', 'quality'], axis=1)
y = df['quality']


# In[267]:


#Convierte la variable categórica Type en variable numérica.
X['Type'] = (X['Type'] == 'red').astype(int)


# In[268]:


#Separa los datos en conjuntos de entrenamiento y conjunto de pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


# In[269]:


#Realiza un escalamiento de los datos para asegurarte que todas las variables numéricas estén en los mismos rangos de valores.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[644]:


#Define el modelo usando
nn_model = MLPRegressor(hidden_layer_sizes=(6,6,6), max_iter=500, random_state=42)


# In[645]:


#Ajusta el modelo a los datos de entrenamiento usando la función
nn_model.fit(X_train_scaled, y_train)


# In[646]:


#Calcula las predicciones del modelo usando
nn_pred = nn_model.predict(X_test_scaled)


# In[647]:


#Evalúa el modelo usando error cuadrado medio, MSE y R-cuadrada
nn_mse = mean_squared_error(y_test, nn_pred)
nn_r2 = r2_score(y_test, nn_pred)


# In[648]:


print("Resultados de la Red Neuronal:")
print(f"Error Cuadrático Medio: {nn_mse}")
print(f"R-cuadrado: {nn_r2}")


# In[649]:


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


# In[12]:


#Comparar los resultados
print('\n Comparación:')
print(f'Mejora en MSE: {(lr_mse - nn_mse) / lr_mse * 100:.2f}%')
print(f'Mejora en R-cuadrado: {(nn_r2 - lr_r2) / lr_r2 *100:.2f}%')


# In[15]:


# Hacer predicciones con la regresión lineal
lr_pred = lr_model.predict(X_test_scaled)
# Comparar las predicciones con los valores reales
resultados = pd.DataFrame({'Real': y_test.values, 'Predicción': lr_pred})
print(resultados.head())  # Mostrar las primeras filas
# Calcular métricas de evaluación
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

print(f"Error Cuadrático Medio (MSE): {lr_mse}")
print(f"Coeficiente de determinación (R²): {lr_r2}")
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(y_test, lr_pred, alpha=0.5)
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("Comparación entre Valores Reales y Predichos")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="dashed")  # Línea de referencia
plt.show()


# In[16]:


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




