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

    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10,) - 0.5

    w3 = np.random.rand(2, 10) - 0.5
    b3 = np.random.rand(2,) - 0.5
    
    return w1, b1, w2, b2, w3, b3

def ReLU(z):
    return np.maximum(0, z)

def ReLU_abl(z):
    return z > 0

def softmax(z):                     #softmax berechnet wahrscheinlichkeiten
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)



def label_zu_vektor(label):
    if label == "Yes":
        return np.array([1,0])
    else:
        return np.array([0,1])


def vorwärts(w1, b1, w2, b2, w3, b3, X):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = ReLU(z2)
    z3 = w3.dot(a2) + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3


def rückwärts(z1, a1, z2, a2, z3, a3, w3, w2, X, Y):
    y_richtig = label_zu_vektor(Y)

    dz3 = a3 - y_richtig
    dw3 = 1/2 * dz3.dot(a2.T)
    db3 = 1/2 * np.sum(dz3, 2)

    dz2 = w3.dot(dz3) * ReLU_abl(z2)
    dw2 = 1/2 * dz2.dot(a1.T)
    db2 = 1/2 * np.sum(dz2, 2)

    dz1 = w2.dot(dz2) * ReLU_abl(z1)
    dw1 = 1/2 * dz1.dot(X.T)
    db1 = 1/2 * np.sum(dz1, 2)
    #rückwärts propagation machen :(
    return dw1, db1, dw2, db2, dw3, db3

    
w1, b1, w2, b2, w3, b3 = init_parameter()
X = features_train[1]
z1, a1, z2, a2, z3, a3 = vorwärts(w1, b1, w2, b2, w3, b3, X)
dw1, db1, dw2, db2, dw3, db3 = rückwärts(z1, a1, z2, a2, z3, a3, w3, w2, X, labels_train[1])

print(dw1, db1, dw2, db2, dw3, db3)

