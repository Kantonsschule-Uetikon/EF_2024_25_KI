import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# 1. Daten laden
# Annahme: Ihr Dataset ist in einer CSV-Datei gespeichert
data = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')  # Ersetzen Sie 'car_data.csv' mit dem Dateinamen Ihres Datensatzes

# 2. Daten bereinigen (falls notwendig)
# Entfernen Sie fehlende Werte oder irrelevante Spalten
data.dropna(inplace=True)

# 3. Features und Zielvariable definieren
X = data.drop(columns=['selling_price', 'name'])  # Features (alle Spalten außer 'selling_price' und 'name')
y = data['selling_price']  # Zielvariable (Verkaufspreis)

# 4. Kategorische und numerische Features identifizieren
categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
numerical_features = ['year', 'km_driven']

# 5. Preprocessing-Pipeline erstellen
# OneHotEncoder für kategorische Features und StandardScaler für numerische Features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ])

# 6. Daten in Trainings- und Testsets aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Preprocessing auf die Daten anwenden
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# 8. Modell definieren
model = Sequential()
model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))  # Eingabeschicht
model.add(Dropout(0.2))  # Dropout zur Regularisierung
model.add(Dense(32, activation='relu'))  # Versteckte Schicht
model.add(Dense(1))  # Ausgabeschicht (Regression, daher keine Aktivierungsfunktion)

# 9. Modell kompilieren
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # MAE: Mean Absolute Error

# 10. Modell trainieren
history = model.fit(X_train, y_train, epochs=50, batch_size=100, validation_split=0.2)

# 11. Modell evaluieren
loss, mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test MAE: {mae}')

# 12. Vorhersagen machen
predictions = model.predict(X_test)
print(predictions[:5])  # Zeige die ersten 5 Vorhersagen