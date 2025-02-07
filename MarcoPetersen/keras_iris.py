import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer




data = pd.read_csv("housing.csv")

X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]
   
# Define price categories
bins = [0, 250000, 500000, 750000, np.inf]
labels = list(range(len(bins) - 1))
y_binned = pd.cut(y, bins=bins, labels=labels)


# Encode the binned categories
y_encoded = LabelEncoder().fit_transform(y_binned)


# Festlegen, welche Spalten numerisch und welche kategorial sind
numerical_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
categorical_features = ['ocean_proximity']


# Preprocessing: Skalierung für numerische Daten und One-Hot-Encoding für kategoriale Daten
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
X_preprocessed = preprocessor.fit_transform(X)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_encoded, test_size=0.5, random_state=42)
print(y_train)

input_shape = X_train.shape[1]  # Anzahl der Features nach dem Preprocessing
output_shape = len(np.unique(y_train)) # Anzahl der Klassen


# Define the model

model = Sequential()
model.add(Dense(64, activation='relu', input_shape = (input_shape,)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(output_shape, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
