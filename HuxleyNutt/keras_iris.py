import pandas as pd
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras.optimizers import SGD
from keras_tuner import RandomSearch
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from keras.callbacks import EarlyStopping

# Load dataset
df = pd.read_csv(r'C:\Users\huxnu\OneDrive\Documents\GitHub\EF_2024_25_KI\HuxleyNutt\poker-hand-training.csv')

# Defining the input features and target variable
X = df.drop('Poker Hand', axis=1)
y = df['Poker Hand']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y.values.reshape(-1, 1))  # Ensure correct shape

# Set n_neighbors to 2 or 3 for smaller classes
smote = SMOTE(sampling_strategy='auto', k_neighbors=2)  # Reduced from 5 to 2
smote_enn = SMOTEENN(smote=smote, random_state=42)

X_resampled, y_resampled = smote_enn.fit_resample(X_scaled, y_one_hot)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.88, random_state=42)

def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(10,)))
    model.add(Dense(hp.Int('units_1', 32, 128, step=32), activation='relu'))
    model.add(Dropout(hp.Float('dropout_1', 0.1, 0.5, step=0.1)))
    model.add(Dense(hp.Int('units_2', 16, 64, step=16), activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=5, executions_per_trial=1)
tuner.search(X_train, y_train, epochs=20, validation_split=0.2)
best_hps = tuner.get_best_hyperparameters()[0]

# Define the model
model = Sequential([
    Input(shape=(10,)),  
    Dense(128, activation='relu'),
    Dropout(0.3),  
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')  # 10 output classes
])

# Compile the model
optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')