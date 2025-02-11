import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

data = pd.read_csv('datensatz.csv')  # Datensatz mit Pandas einlesen
teams = list(set(data['home_team']))  # eindeutige Liste aller Teams, home_team und away_team enthalten die gleichen Teams, also muss nur eine der beiden verwendet werden

# One-hot encode the target variables
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') #One-Hot-Encoder wird initialisiert
encoder.fit([[team] for team in teams]) #One-Hot-Encoder wird auf die Teams gefittet

home_encoded = encoder.transform(data[['home_team']])
away_encoded = encoder.transform(data[['away_team']])  #Die Teams werden in numerische Werte umgewandelt
X = np.hstack([home_encoded, away_encoded])  #Die numerischen Werte werden in ein Array zusammengefügt

label_encoder = LabelEncoder() #Label-Encoder wird initialisiert
data['result'] = label_encoder.fit_transform(data['result'])  # kategorische Werte werden in numerische Werte umgewandelt
y_one_hot = pd.get_dummies(data['result']).values  #Die numerischen Werte werden mit Pandas in One-Hot-Encoded Werte umgewandelt

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)  # Split the data into training and testing sets

# Define the model
model = Sequential()
model.add(Input(shape=(X_train.shape[1],))) #Input-Layer wird hinzugefügt, X_train.shape[1] gibt die Anzahl Spalten der Matrix X_train an, welches der Anzahl Features entspricht
model.add(Dense(64, activation='relu'))  #Hidden-Layer wird hinzugefügt, 64 Neuronen
model.add(Dropout(0.2)) #Dropout-Layer wird hinzugefügt, 20% der Neuronen werden deaktiviert
model.add(Dense(y_train.shape[1], activation='softmax')) #Output-Layer wird hinzugefügt, y_train.shape[1] gibt die Anzahl Spalten der Matrix y_train an, welches der Anzahl Labels entspricht

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #Das Modell wird kompiliert

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}') 

# Function to predict a match outcome
def predict_match(home_team, away_team): #Die Funktion predict_match nimmt die Namen der beiden Teams als Input
    # Convert teams to one-hot encoded format
    home_encoded = encoder.transform([[home_team]]) 
    away_encoded = encoder.transform([[away_team]]) #Die gewählten Teams werden one-hot-encodiert
    match_data = np.hstack([home_encoded, away_encoded]) #Die one-hot-encodierten Teams werden in ein Array zusammengefügt
    
    prediction = model.predict(match_data) #Das Modell wird auf die Daten angewendet und bestimmt die Wahrscheinlichkeiten für die verschiedenen Ergebnisse
    # Format output
    result_labels = label_encoder.classes_  # ['Away Win', 'Draw', 'Home Win']
    probabilities = {result_labels[i]: round(prediction[0][i] * 100, 2) for i in range(len(result_labels))} #Die Wahrscheinlichkeiten werden in Prozent umgewandelt
    print(f"Predicted Probabilities:\n{probabilities}") #Die Wahrscheinlichkeiten werden ausgegeben
    print(f"Most Likely Outcome: {max(probabilities, key=probabilities.get)}") #Das wahrscheinlichste Ergebnis wird ausgegeben

# Example usage
predict_match(str(input("Home Team:")), str(input("Away Team:"))) #Die Funktion predict_match wird aufgerufen und die Namen der Teams werden als Input übergeben