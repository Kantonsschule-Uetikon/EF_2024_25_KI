
import pandas as pd
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from keras.optimizers import SGD

# Load dataset
df = pd.read_csv(r'C:\Users\huxnu\OneDrive\Documents\python\poker-hand-testing.csv')

# Defining the input features and target variable
X = df.drop('Poker Hand', axis=1)
y = df['Poker Hand']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# One-hot encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y.values.reshape(-1, 1))  # Ensure correct shape

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_one_hot, test_size=0.99, random_state=42)

# Define the model
model = Sequential([
    Input(shape=(10,)),  
    Dense(64, activation='relu'),  # Increased neurons
    Dropout(0.3),  # Stronger regularization
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes
])

# Compile the model
optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


from keras_tuner import RandomSearch

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


# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')




