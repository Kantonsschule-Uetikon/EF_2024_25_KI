import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# Dataset laden
raw_data = np.genfromtxt("japan_heart_attack_dataset.csv", delimiter=",", dtype=str, skip_header=1)
# zu löschende Spalten definieren
columns_to_delete = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
# nichtgewollte Spalten löschen
data = np.delete(raw_data, columns_to_delete, axis=1)

# Features und Zielvariable trennen
X = data[:, :-1]  # Features
y = data[:, -1]  # Target variable

for column in range(data.shape[1]):
    le = LabelEncoder()
    X[:, column] = le.fit_transform(X[:, column])

X = X.astype(float)

# One-hot encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Input(shape=(X_train.shape[1],))) 
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(y_one_hot.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
