import numpy as np
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.preprocessing import OneHotEncoder, LabelEncoder #type: ignore

# Load the dataset
raw_data = np.genfromtxt("japan_heart_attack_dataset.csv", delimiter=",", dtype=str, skip_header=1)

# Specify the indices of the columns you want to delete
columns_to_delete = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

# Delete the specified columns
data = np.delete(raw_data, columns_to_delete, axis=1)

# Separate features and target variable
X = data[:, :-1]  # Features
y = data[:, -1]  # Target variable

# Identify categorical columns (assuming all columns with string data are categorical)
categorical_columns = np.where(X[0, :] == X[0, :].astype(str))[0]

# Encode categorical features
for col in categorical_columns:
    le = LabelEncoder()
    X[:, col] = le.fit_transform(X[:, col])

# Convert features to float
X = X.astype(float)

# One-hot encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))  # Ensure input shape matches the number of features
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(y_one_hot.shape[1], activation='softmax'))  # Ensure output neurons match the number of classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')