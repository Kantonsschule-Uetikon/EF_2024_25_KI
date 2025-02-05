import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


DATAFILE = "Social_Network_Ads.csv"

def einlesen(filename):
    features = np.loadtxt(filename, dtype=np.int_, usecols=[0,1], skiprows=1, delimiter=",")
    resultate = np.loadtxt(filename, dtype=np.bool_, usecols=[2], skiprows=1, delimiter=",")
    return features, resultate

features, resultate = einlesen(DATAFILE)

X = features
y = resultate

encoder = OneHotEncoder(sparse_output=False)
y_one_hot = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(500, activation='relu', input_dim=2))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20)

pred_train= model.predict(X_train)
scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
 
pred_test= model.predict(X_test)
scores2 = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))