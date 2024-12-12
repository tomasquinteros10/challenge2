import pandas as pd

data = pd.read_csv('dataset_flujo_vehicular.csv') 

 #primeras filas
print(data.head())  

#Información general sobre el dataset
print(data.info())   
print(data.describe()) 

# cantidad de filas 
cantidad_filas = len(data)
print(f"Cantidad de filas: {cantidad_filas}")

# valores nulos de cada columna
print(f"valores nulos en cada columna: \n{data.isnull().sum()}")

# cantidad de filas duplicadas
print(f"filas duplicadas: {data.duplicated().sum()}")


# cambio de datetime 
data['HORA'] = pd.to_datetime(data['HORA'], errors='coerce', dayfirst=True)

# verificacion de la conversion
print(data['HORA'].head())

# valores no convertidos
nulos_hora = data['HORA'].isnull().sum()
print(f'valores nulos después de la conversión: {nulos_hora}')

# normalizar la columna SENTIDO
data['SENTIDO'] = data['SENTIDO'].str.lower()

# verificar los valores únicos de columna SENTIDO
print(f"valores únicos de SENTIDO: {data['SENTIDO'].unique()}")

# calcular cuartiles y rango intercuartil 
Q1 = data['CANTIDAD'].quantile(0.25)
Q3 = data['CANTIDAD'].quantile(0.75)
IQR = Q3 - Q1
# 186412 10123
# identificacion de valores atipicos
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR
atypicall = data[(data['CANTIDAD'] < limite_inferior) | (data['CANTIDAD'] > limite_superior)]
cantidad_filas2 = len(atypicall)
print(f"filas con datos atip{cantidad_filas2}")
print(f"Los valores atipicos son: \n{atypicall}")

# Imputar outliers con la mediana
import numpy as np
mediana = data['CANTIDAD'].median()
data['CANTIDAD'] = np.where((data['CANTIDAD'] < limite_inferior) | (data['CANTIDAD'] > limite_superior), mediana, data['CANTIDAD'])
data2 = data
print(data2)
# Verificar el número de outliers restantes
outliers = data2[(data2['CANTIDAD'] < limite_inferior) | (data2['CANTIDAD'] > limite_superior)]
print(f'Número de outliers restantes: {outliers.shape[0]}')

# verificar la latitud y la longitud
invalid_latitudes = data[(data['LATITUD'] < -90) | (data['LATITUD'] > 90)]
invalid_longitudes = data[(data['LONGITUD'] < -180) | (data['LONGITUD'] > 180)]
print(f"Número de latitudes inválidas: {len(invalid_latitudes)}")
print(f"Número de longitudes inválidas: {len(invalid_longitudes)}")

# suma de elementos de cantidad
total_cantidad= data['CANTIDAD'].sum()
print(f"suma de los valores de 'cantidad': {total_cantidad}")

# data_filtered = [x for x in data['CANTIDAD'] if x < 20000] 
#print(len(data['CANTIDAD']))
# print(len(data_filtered))       
# data_filtered2 = [x for x in data['CANTIDAD'] if x >=20000]
# print(len(data_filtered2))
# print(data_filtered)
#import matplotlib.pyplot as plt
#Si hay una columna numérica, graficamos su distribución
# plt.hist(data2,bins=20)
# plt.title('CANTIDAD')
# plt.xlabel('Valor')
# plt.ylabel('Frecuencia')
# plt.show()


print(f'datos actualizados: \n{data2}')
 
# nuevas componentes de la columna HORA
data2['HORA'] = pd.to_datetime(data2['HORA'])
data2['año'] = data2['HORA'].dt.year
data2['mes'] = data2['HORA'].dt.month
data2['dia'] = data2['HORA'].dt.day
data2['hora'] = data2['HORA'].dt.hour
print(data2['hora'].head())

# densidad de tráfico (cantidad dividida por diferencia absoluta de coordenadas)
data2['densidad'] = data2['CANTIDAD'] / (data2['LATITUD'] - data2['LONGITUD']).abs()

# # escalar la columna CANTIDAD
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# data2['CANTIDAD_ESCALADA'] = scaler.fit_transform(data2[['CANTIDAD']])

# diccionario para codificar las categorías
codificacion = {'interna': 0, 'ingreso': 1, 'egreso': 2}
data2['SENTIDO_CODIFICADO'] = data2['SENTIDO'].map(codificacion)
print(data2[['SENTIDO', 'SENTIDO_CODIFICADO']].head())
print(data2.head())
# Después de transformar los datos, puedes realizar análisis agrupados para obtener insights sobre el flujo vehicular según el sentido:
# agrupar por sentido y sumar la cantidad
analisis_sentido = data2.groupby('SENTIDO')['CANTIDAD'].sum()
print(f"sentido por cantidad: \n{analisis_sentido}")
print(data2['SENTIDO_CODIFICADO'])
import matplotlib.pyplot as plt

# cantidad por hora
cantidadpor_hora = data2.groupby('hora')['CANTIDAD'].sum()
print(f"promedio por hora: \n{cantidadpor_hora}")
cantidadpor_hora.plot(kind='bar', color='skyblue', figsize=(8, 6))

# Personalizar el gráfico
plt.title(' vehiculos por hora')
plt.xlabel('hora')
plt.ylabel('vehiculo')
plt.xticks(rotation=45)
plt.show()

# agrupar cantidad por coordenadas
flujo_por_coordenadas = data2.groupby(['LATITUD', 'LONGITUD'])['CANTIDAD'].sum()
print(f"flujo de cantidad por coordenadas: \n{flujo_por_coordenadas}")    

            # tabla dinámica
tabla_dinamica = data2.pivot_table(values='CANTIDAD', index='HORA', columns='SENTIDO', aggfunc='sum', fill_value=0)
print(f" la tabla dinamica: \n{tabla_dinamica}")

 # promedio por 'SENTIDO'
promedio_por_sentido = data2. pivot_table(values='CANTIDAD', index='SENTIDO', aggfunc='mean')
print(f" el promedio de datos por sentido: \n{promedio_por_sentido}")
    
#     # se pueden hacer muchas tablas 
# # filtro por tipo de dato -> 
# egreso_data = data2['SENTIDO'] == 'egreso'
# ingreso_data= data2['SENTIDO'] == 'ingreso'
# interna_data= data2['SENTIDO'] == 'interna'
# print(egreso_data.head())
# print()
# print(ingreso_data.head())
# print()
# print(interna_data.head())
# print()

# primer y ultimo dia del dataset
min_dia=data2['dia'][(data2['año']==2020) & (data2['mes']==3)].min()
print(f"el primer dia es {min_dia} de marzo de 2020")
max_dia=data2['dia'][(data2['año']==2022) & (data2['mes']==3)].max()
print(f"el ultimo dia es {max_dia} de marzo de 2023")

# cantidad de vehiculos en circulacion el dia 1 de mayo de 2020 y 2021
el1_mayo2020 = data2['CANTIDAD'][(data2['dia']  == 1 )& (data2['mes'] == 5) & (data2['año'] == 2020) ].sum()
print(f"la cantidad total de vehiculos en 1/5/2020 es  {el1_mayo2020}")
el1_mayo2021 = data2['CANTIDAD'][(data2['dia']  == 1 )& (data2['mes'] == 5) & (data2['año'] == 2021) ].sum()
print(f"la cantidad total de vehiculos en 1/5/2021 es  {el1_mayo2021}")

# cantidad de vehiculos en inicio de pandemia 20/3 de 2020 y 2021 y en 12/3 de 2022 en hora pico de CABA
horapico1 = data2['CANTIDAD'][(data2['hora'] >= 7 )& (data2['hora'] <= 11) & (data2['año'] == 2020)&(data2['dia'] ==20) &(data2['mes']== 3)].sum()
print(f"horapico2020: {horapico1}")
horapico2 = data2['CANTIDAD'][(data2['hora'] >= 7 )& (data2['hora'] <= 11) & (data2['año'] == 2021)&(data2['dia'] ==20) &(data2['mes']== 3)].sum()
print(f"horapico2021: {horapico2}") 
horapico3 = data2['CANTIDAD'][(data2['hora'] >= 7 )& (data2['hora'] <= 11) & (data2['año'] == 2022)&(data2['dia'] ==12) &(data2['mes']== 3)].sum()
print(f"horapico2022: {horapico3}")
horapico1_0 = data2['CANTIDAD'][(data2['hora'] >= 16 )& (data2['hora'] <= 20) & (data2['año'] == 2020)&(data2['dia'] ==2) &(data2['mes']== 3)].sum()
print(f"horapico2020 n: {horapico1_0}")
horapico2_0 = data2['CANTIDAD'][(data2['hora'] >= 16 )& (data2['hora'] <= 20) & (data2['año'] == 2021)&(data2['dia'] ==2) &(data2['mes']== 3)].sum()
print(f"horapico2021 n: {horapico2_0}")
horapico3_0 = data2['CANTIDAD'][(data2['hora'] >= 16 )& (data2['hora'] <= 20) & (data2['año'] == 2022)&(data2['dia'] ==2) &(data2['mes']== 3)].sum()
print(f"horapico2022 n: {horapico3_0}")

# comparacion de 2 de marzo en rango de 7 a 20
cant1 = data2['CANTIDAD'][(data2['hora'] >= 7 )& (data2['hora'] <= 20) & (data2['año'] == 2020)&(data2['dia'] ==2) &(data2['mes']== 3)].sum()

print(f"cantidad en 2/3/2020: {cant1}")

cant2 = data2['CANTIDAD'][(data2['hora'] >= 7) & (data2['hora'] <= 20) & (data2['año'] == 2021)&(data2['dia'] ==2) &(data2['mes']== 3)].sum()

print(f"cantidad en 2/3/2021: {cant2}")

cant3 = data2['CANTIDAD'][(data2['hora'] >= 7 )& (data2['hora'] <= 20) & (data2['año'] == 2022)&(data2['dia'] ==2) &(data2['mes']== 3)].sum()

print(f"cantidad en 2/3/2022: {cant3}")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# variables categóricas a numéricas
label_encoders = {}
for column in ['CODIGO_LOCACION', 'HORA', 'SENTIDO']:
    le = LabelEncoder()
    data2[column] = le.fit_transform(data2[column])
    label_encoders[column] = le  # Guardar el encoder si se necesita más tarde

# Definir las características (X) y la variable objetivo (y)
X = data2[['CODIGO_LOCACION', 'HORA', 'SENTIDO', 'LATITUD', 'LONGITUD']]
y = data2['CANTIDAD']

# Dividir los datos en conjuntos de entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Visualizar las primeras filas del DataFrame
print(data2.head())

# Convertir variables categóricas a numéricas
label_encoders = {}
for column in ['CODIGO_LOCACION', 'HORA', 'SENTIDO']:
    le = LabelEncoder()
    data2[column] = le.fit_transform(data2[column])
    label_encoders[column] = le  # Guardar el encoder si se necesita más tarde

# Definir las características (X) y la variable objetivo (y)
X = data2[['CODIGO_LOCACION', 'HORA', 'SENTIDO', 'LATITUD', 'LONGITUD']]
y = data2['CANTIDAD']

# Dividir los datos en conjuntos de entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de Árbol de Decisión
model = DecisionTreeRegressor(random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')
