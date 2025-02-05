import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


#Pre Processing
marriage_type_map = {"Love": -1, "Arranged": 1}
urban_rural_map = {"Rural": -1, "Urban": 1}
family_involvement_map = {"Low": 1, "Moderate": 2, "High": 3}
divorce_status_map = {"No": -1, "Yes": 1}

daten_unbearbeitet = np.genfromtxt("marriage_divorce_india_with_id.csv", delimiter=",", dtype=str, skip_header=1)

daten  = np.array([
    [
        float(row[1]),  # Marriage Duration
        float(row[2]),  # Age at Marriage
        marriage_type_map[row[3]],  # Marriage Type
        #float(row[5]),  # Income
        urban_rural_map[row[7]],  # Urban/Rural
        family_involvement_map[row[8]],  # Family Involvement
        float(row[9]),  # Children
    ]
    for row in daten_unbearbeitet], dtype=np.float64)

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(daten_unbearbeitet[:,10])

features_ev = daten.astype(float)[0:200] #_ev für das spätere testen
features_train = daten.astype(float)[200:1200]

labels_ev = labels[0:200]
labels_train = labels[200:1200]

"""
encoder = OneHotEncoder(sparse_output=False)
labels_train_one_hot = encoder.fit_transform(labels_train.reshape(-1, 1))
labels_ev_one_hot = encoder.transform(labels_ev.reshape(-1, 1))
"""

model = Sequential()
model.add(Input(shape=(6,)))
model.add(Dense(100, activation="sigmoid"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", "recall", "precision"])

model.fit(features_train, labels_train, epochs=10, batch_size=8, validation_split=0.2)

loss, accuracy, recall, precision = model.evaluate(features_ev, labels_ev)
print(f'Loss: {loss}, Accuracy: {accuracy}, {recall=}, {precision=}')

testlabels = model.predict(features_ev)
print(testlabels)
