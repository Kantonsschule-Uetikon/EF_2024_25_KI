import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

data = pd.read_csv("Häuser.csv")

X = data.drop("Price", axis=1)
y = data["Price"]


# One-hot encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

# Festlegen, welche Spalten numerisch und welche kategorial sind
numerical_features = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt']
categorical_features = ['Location', 'Condition', 'Garage']


# Preprocessing: Skalierung für numerische Daten und One-Hot-Encoding für kategoriale Daten
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
X_preprocessed = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_one_hot, test_size=0.2, random_state=42)

input_shape = X_train.shape[1]  # Anzahl der Features nach dem Preprocessing
output_shape = y_train.shape[1]

# Define the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape = (input_shape,)))
model.add(Dropout(0.1))
model.add(Dense(output_shape, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
