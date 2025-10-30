import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import models, layers
from sklearn.preprocessing import OneHotEncoder


data = np.load('rps_data.npy')
labels = np.load('rps_labels.npy')

# Crear nombres de columnas
columns = []
for i in range(1, 22):
    columns.append(f'x{i}')
    columns.append(f'y{i}')

# Crear DataFrame
df = pd.DataFrame(data, columns=columns)
df['label'] = labels

print(df.head())
print(df.info())
print(df.describe())

# Separar X e y
X = df.drop('label', axis=1).values  # coordenadas de landmarks
y = df['label'].values                # etiquetas 0=piedra,1=papel,2=tijeras


y = y.reshape(-1,1)
# Crear encoder con nueva sintaxis
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)


#  Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y 
)


# Crear el modelo
model = models.Sequential([
    layers.Input(shape=(42,)),                 
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),                       
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),                       
    layers.Dense(3, activation='softmax')      
])

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar modelo
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16
)

# Evaluar en train
loss_train, acc_train = model.evaluate(X_train, y_train, verbose=0)
print(f"Accuracy en train: {acc_train*100:.2f}%")

# Evaluar en test
loss_test, acc_test = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy en test: {acc_test*100:.2f}%")

# Guardar el modelo entrenado
model.save('rps_model.h5')


print("Modelo  guardado correctamente.")
