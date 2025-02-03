import numpy as np
import pandas as pd
import sklearn
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

data = pd.read_csv('datensatz.csv') # Datensatz mit Pandas einlesen
teams = list(set(data['home_team'])) # eindeutige Liste aller Teams, home_team und away_team enthalten die gleichen Teams, also muss nur eine der beiden verwendet werden
label_encoder = LabelEncoder()
data['result'] = label_encoder.fit_transform(data['result'])  # kategorische Werte werden in numerische Werte umgewandelt
X =                         # Matrix mit Features × Datensätze
y =        # Labels


# One-hot encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Input(shape=(4,)))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
print(sklearn.__version__)