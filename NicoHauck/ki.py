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
        float(row[5]),  # Income Level
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

def init_parameter():
    w1 = np.random.rand(10, 7) - 0.5
    b1 = np.random.rand(10,) - 0.5

    w2 = np.random.rand(2, 10) - 0.5
    b2 = np.random.rand(2,) - 0.5
    
    return w1, b1, w2, b2

def aktivierung(z):
    if z > 0:
        return 1
    else:
        return 0

def softmax(z):                      #softmax gibt wahrscheinlichkeiten zurück
    return np.exp(z) / sum(np.exp(z))

def vorwärts(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1
    a1 = aktivierung(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2



