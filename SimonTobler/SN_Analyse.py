import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


DATAFILE = "Social_Network_Ads.csv"

def einlesen(filename):
    features = np.loadtxt(filename, dtype=np.float64, usecols=[0,1], skiprows=1, delimiter=",")
    resultate = np.loadtxt(filename, dtype=np.float64, usecols=[2], skiprows=1, delimiter=",")
    return features, resultate

features, resultate = einlesen(DATAFILE)

X = features
y = resultate

featureScaler = StandardScaler()
X = featureScaler.fit_transform(X)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential()
model.add(Input((2,)))
model.add(Dense(18, activation='relu'))
#model.add(Dense(100, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50)

print(y_train)

pred_train= model.predict(X_train)
scores = model.evaluate(X_train, y_train, verbose=0)
print('Genauigkeit mit Trainingsdaten: {} \n Fehlerquote mit Trainingsdaten: {}'.format(scores[1], 1 - scores[1]))   
 
pred_test= model.predict(X_test)
scores2 = model.evaluate(X_test, y_test, verbose=0)
print('Genauigkeit mit Testdaten: {} \n Fehlerquote mit Testdaten: {}'.format(scores2[1], 1 - scores2[1]))