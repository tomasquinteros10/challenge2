import pandas as pd

# se carga el archivo CSV
data = pd.read_csv('dataset_flujo_vehicular.csv')

# # Explorar las primeras filas para confirmar la carga
# print(data.head())
# convierto los datos de cantidad en int (ESTAN EN FLOAT)
data['CANTIDAD'].astype(int)
# # Información general sobre el dataset
# #print(data.info())

# # Resumen estadístico de las columnas numéricas
# print(data.describe())

#cambio de datetime 
# Convertir la columna HORA a tipo datetime, permitiendo inferir el formato
data['HORA'] = pd.to_datetime(data['HORA'], errors='coerce', dayfirst=True)

# Verificar la conversión
print(data['HORA'].head())
print(data['HORA'].dtype)  # Debe mostrar datetime64[ns]
# Verificar cuántos valores no se pudieron convertir
nulos_hora = data['HORA'].isnull().sum()
print(f'Número de valores nulos después de la conversión: {nulos_hora}')

# Opcionalmente, puedes filtrar y ver los valores originales que no se pudieron convertir
valores_no_convertibles = data[data['HORA'].isnull()]
print(valores_no_convertibles)

# Normalizar la columna SENTIDO a minúsculas
data['SENTIDO'] = data['SENTIDO'].str.lower()
# Verificar los valores únicos en la columna SENTIDO
print(data['SENTIDO'].unique())

# Verificar cuántos valores nulos hay en cada columna
print(f"\nLos valores nulos en cada columna: {data.isnull().sum()}")

# Calcular Q1 y Q3
Q1 = data['CANTIDAD'].quantile(0.25)
Q3 = data['CANTIDAD'].quantile(0.75)
IQR = Q3 - Q1

# Calcular límites
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Identificar valores atípicos
atypicall = data[(data['CANTIDAD'] < limite_inferior) | (data['CANTIDAD'] > limite_superior)]

print("Valores atípicos encontrados: ")
print(atypicall) #no hay xd

# Verificar latitud y longitud
invalid_latitudes = data[(data['LATITUD'] < -90) | (data['LATITUD'] > 90)]
invalid_longitudes = data[(data['LONGITUD'] < -180) | (data['LONGITUD'] > 180)]

print(f"Número de latitudes inválidas: {len(invalid_latitudes)}")
print(f"Número de longitudes inválidas: {len(invalid_longitudes)}")

# Número de filas duplicadas
print(f"\nNúmero de filas duplicadas: {data.duplicated().sum()}")

# #cantidad de filas 
# cantidad_filas = len(data)
# print(f"Cantidad de filas: {cantidad_filas}")

# # suma de elementos de cantidad
# total_cantidad= data['CANTIDAD'].sum()
# print(f"La suma de todos los autos es: {total_cantidad}")

# #primera fila 
# primera_fila = data.head(1)
# print(primera_fila)

# #ultima fila
# ultima_fila = data.tail(1)
# print(ultima_fila)

#imprimir valor menor y mayor
#print(f"El minimo es: {data['CANTIDAD'].min()}") 
# print(f"El maximo es: {data['CANTIDAD'].max()}") a filtred
# data_filtered = [x for x in data['CANTIDAD'] if x < 20000] 
#print(len(data['CANTIDAD']))
# print(len(data_filtered))       
# data_filtered2 = [x for x in data['CANTIDAD'] if x >=20000]
# print(len(data_filtered2))
# print(data_filtered)

#import matplotlib.pyplot as plt

# #Si hay una columna numérica, graficamos su distribución
# plt.hist(data_filtered,bins=20)
# plt.title('CANTIDAD')
# plt.xlabel('Valor')
# plt.ylabel('Frecuencia')
# plt.show()

#verifico cantidad de valores nulos
print(data[['LATITUD', 'LONGITUD']].isnull().sum())
#Decisión sobre Eliminación: Un porcentaje del 1.79% es relativamente bajo. Si decides eliminar estas filas, el impacto en el conjunto de datos será mínimo.(nulos/nfilas)*100)
# Eliminar filas con valores nulos en las columnas LATITUD y LONGITUD
data_cleaned = data.dropna(subset=['LATITUD', 'LONGITUD'])

# Verificar la cantidad de valores nulos después de la eliminación
print("\nValores nulos después de la eliminación:")
print(data_cleaned.isnull().sum())

# Mostrar el número de filas antes y después
print(f'\n Número de filas originales: {data.shape[0]}')
print(f'\n Número de filas después de eliminar: {data_cleaned.shape[0]}')
data_cleaned.to_csv('dataset_flujo_vehicular.csv', index=False)
print('datos actualizados')
 
# 1. Extraer componentes de la columna HORA
data_cleaned['HORA'] = pd.to_datetime(data_cleaned['HORA'])

# Crear nuevas columnas: año, mes, día, hora
data_cleaned['año solo'] = data_cleaned['HORA'].dt.year
data_cleaned['mes solo'] = data_cleaned['HORA'].dt.month
data_cleaned['día solo'] = data_cleaned['HORA'].dt.day
data_cleaned['hora solo'] = data_cleaned['HORA'].dt.hour
 
print(data_cleaned['hora solo'].head())
# 2. Calcular densidad de tráfico (cantidad dividida por diferencia absoluta de coordenadas)
data_cleaned['densidad'] = data_cleaned['CANTIDAD'] / (data_cleaned['LATITUD'] - data_cleaned['LONGITUD']).abs()


# 3. Escalar la columna CANTIDAD
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_cleaned['CANTIDAD_ESCALADA'] = scaler.fit_transform(data_cleaned[['CANTIDAD']])

# Crear un diccionario para codificar las categorías
codificacion = {'interna': 0, 'ingreso': 1, 'egreso': 2}

# Aplicar la codificación
data_cleaned['SENTIDO_CODIFICADO'] = data_cleaned['SENTIDO'].map(codificacion)

# Verificar los resultados de la codificacion
print(data_cleaned[['SENTIDO', 'SENTIDO_CODIFICADO']].head())

# Después de transformar los datos, puedes realizar análisis agrupados para obtener insights sobre el flujo vehicular según el sentido:

# Agrupar por sentido y sumar la cantidad
analisis_sentido = data_cleaned.groupby('SENTIDO')['CANTIDAD'].sum()
print(analisis_sentido)

# Agrupar cantidad por hora
promedio_por_hora = data_cleaned.groupby('HORA')['CANTIDAD'].mean()
print(promedio_por_hora)

# Agrupar cantidad por coordenadas
flujo_por_coordenadas = data.groupby(['LATITUD', 'LONGITUD'])['CANTIDAD'].sum()
print(flujo_por_coordenadas)    

#Las tablas dinámicas son una forma poderosa de resumir datos.
# Crear una tabla dinámica
tabla_dinamica = data.pivot_table(values='CANTIDAD', index='HORA', columns='SENTIDO', aggfunc='sum', fill_value=0)
print(f"\n LA TABLA DINAMICA: {tabla_dinamica}")

# Promedio por 'SENTIDO'
promedio_por_sentido = data.pivot_table(values='CANTIDAD', index='SENTIDO', aggfunc='mean')
print(f"\n EL PROMEDIO DE AUTOS POR SENTIDO {promedio_por_sentido}")
 
#se pueden hacer mas tablas(si)
#filtro por tipo de dato -> 
egreso_data = data[data['SENTIDO'] == 'egreso']
print(egreso_data.head())
filtrado_horas = data_cleaned[(data_cleaned['hora solo'] >= 14) & (data_cleaned ['hora solo'] <= 16)]
print(filtrado_horas.head())

print(data_cleaned)

