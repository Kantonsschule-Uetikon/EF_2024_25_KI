import tensorflow as tf
from tensorflow import keras
import numpy as np

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
        str(row[10])  # Divorce Status
    ]
    for row in daten_unbearbeitet])

features_ev = daten[:, :-1].astype(float)[0:200] #_ev für das spätere testen
features_train = daten[:, :-1].astype(float)[200:1200]

labels_ev = daten[:, -1][0:200]
labels_train = daten[:, -1][200:1200]

