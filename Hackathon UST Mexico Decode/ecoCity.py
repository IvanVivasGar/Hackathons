import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Ruta del archivo CSV
csv_file_path = '/Users/ivanvivasgarcia/Documents/PortafolioPersonal/Hackathons/Hackathon UST Mexico Decode/datasets/Water Consumption 1st Sem 2019.csv'
csv_file_path2 = '/Users/ivanvivasgarcia/Documents/PortafolioPersonal/Hackathons/Hackathon UST Mexico Decode/datasets/ciudadMXClima.csv'

# Leer el archivo CSV
df = pd.read_csv(csv_file_path)
df2 = pd.read_csv(csv_file_path2)

# Convertir las columnas de fecha a datetime
df['date'] = pd.to_datetime(df['fecha_referencia'])
df2['date'] = pd.to_datetime(df2['date'])

# Añadir columna de bimestre
df['bimestre'] = df['date'].dt.month // 2 + 1
df2['bimestre'] = df2['date'].dt.month // 2 + 1

# Inicializar una lista para almacenar los success rates
success_rates = []

# Obtener la lista de colonias únicas
colonias = df['colonia'].unique()

# Procesar cada colonia por separado
for colonia in colonias:
    df_colonia = df[df['colonia'] == colonia]

    # Calcular el promedio de consumo_total por bimestre
    df_avg_bimestre = df_colonia.groupby('bimestre')['consumo_total'].mean().reset_index()

    # Verificar que hay datos para los tres bimestres
    if len(df_avg_bimestre) < 3:
        continue

    # Separar los datos en entrenamiento (bimestres 1 y 2) y prueba (bimestre 3)
    df_train = df_avg_bimestre[df_avg_bimestre['bimestre'].isin([2, 3])]
    df_test = df_avg_bimestre[df_avg_bimestre['bimestre'] == 4]

    # Imprimir los datos de entrenamiento y prueba
    print()
    print(f'Colonia: {colonia}')

    # Realiza un merge de los DataFrames basándote en la columna de bimestre
    merged_train = pd.merge(df_train, df2, on='bimestre')
    merged_test = pd.merge(df_test, df2, on='bimestre')

    # Crear nuevas características
    merged_train['temp_range'] = merged_train['tmax'] - merged_train['tmin']
    merged_test['temp_range'] = merged_test['tmax'] - merged_test['tmin']

    # Seleccionar las características y la variable objetivo
    X_train = merged_train[['tavg', 'tmin', 'tmax', 'wdir', 'wspd', 'pres', 'temp_range']]
    y_train = merged_train['consumo_total']

    # Calcular el promedio de las características climáticas para el bimestre de prueba
    X_test_avg = merged_test[['tavg', 'tmin', 'tmax', 'wdir', 'wspd', 'pres', 'temp_range']].mean().to_frame().T
    y_test_avg = df_test['consumo_total'].mean()

    # Verificar que X_train y X_test_avg no estén vacíos
    if X_train.empty or X_test_avg.empty:
        continue

    # Normalizar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_avg_scaled = scaler.transform(X_test_avg)

    # Crear el modelo de regresión Ridge
    model = Ridge(alpha=1.0)

    # Entrenar el modelo
    model.fit(X_train_scaled, y_train)

    # Realizar una única predicción en el conjunto de prueba
    y_test_pred_avg = model.predict(X_test_avg_scaled)[0]

    # Imprimir la predicción
    print(f'Predicción: {y_test_pred_avg}')

    # Evaluar el modelo en el conjunto de prueba
    test_mse = mean_squared_error([y_test_avg], [y_test_pred_avg])

    # Evaluar el rango aceptable de predicción
    acceptable_range = 50
    within_range = (y_test_pred_avg >= (y_test_avg - acceptable_range)) & (y_test_pred_avg <= (y_test_avg + acceptable_range))
    percentage_within_range = 100 if within_range else 0

    # Agregar el success rate a la lista
    success_rates.append(percentage_within_range)

# Calcular el promedio del success rate
if success_rates:
    average_success_rate = sum(success_rates) / len(success_rates)
    print(f'Promedio del success rate general: {average_success_rate:.2f}%')
else:
    print('No se pudo calcular el promedio del success rate general porque no hay datos válidos.')