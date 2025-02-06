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

def init_parameter():
    w1 = np.random.rand(10, 6) - 0.5
    b1 = np.random.rand(10,) - 0.5

    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10,) - 0.5

    w3 = np.random.rand(2, 10) - 0.5
    b3 = np.random.rand(2,) - 0.5
    
    return w1, b1, w2, b2, w3, b3

def ReLU(z):
    return np.maximum(0, z)

def ReLU_abl(z):
    return (z > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_abl(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def softmax(z):                     #softmax berechnet wahrscheinlichkeiten
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)


def label_zu_vektor(label):
    if label == "Yes":
        return np.array([1,0])
    else:
        return np.array([0,1])

import numpy as np

def a_to_output(a):
    output = np.zeros_like(a)
    output[np.argmax(a)] = 1
    return output.astype(int)


def vorwärts(w1, b1, w2, b2, w3, b3, X):
    z1 = w1.dot(X) + b1
    a1 = sigmoid(z1)

    z2 = w2.dot(a1) + b2
    a2 = sigmoid(z2)

    z3 = w3.dot(a2) + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3


def rückwärts(z1, a1, z2, a2, z3, a3, w3, w2, X, Y):
    
    Y = label_zu_vektor(Y)
    
    # output
    dz3 = a3 - Y  # softmax ableitung anscheinend
    dw3 = 1/2 * np.outer(dz3, a2)
    db3 = 1/2 * dz3
    
    # hidden 2
    dz2 = w3.T.dot(dz3) * sigmoid_abl(z2)
    dw2 = 1/2 * np.outer(dz2, a1)
    db2 = 1/2 * dz2
    
    # hidden 1
    dz1 = w2.T.dot(dz2) * sigmoid_abl(z1)
    dw1 = 1/2 * np.outer(dz1, X)
    db1 = 1/2 * dz1
    
    return dw1, db1, dw2, db2, dw3, db3


def update_parameter(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, learning_rate):
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    w3 -= learning_rate * dw3
    b3 -= learning_rate * db3
    return w1, b1, w2, b2, w3, b3

def trainieren(X, Y, iterationen, learning_rate):
    w1, b1, w2, b2, w3, b3 = init_parameter()
    for j in range(iterationen):    
        for i in range(1000):
            z1, a1, z2, a2, z3, a3 = vorwärts(w1, b1, w2, b2, w3, b3, X[i])
            dw1, db1, dw2, db2, dw3, db3 = rückwärts(z1, a1, z2, a2, z3, a3, w3, w2, X[i], Y[i])
            w1, b1, w2, b2, w3, b3 = update_parameter(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, learning_rate)
    return w1, b1, w2, b2, w3, b3
    
w1, b1, w2, b2, w3, b3 = trainieren(features_train, labels_train, 100, 0.1)



treffer = 0
for i in range(0, 200):
    z1, a1, z2, a2, z3, a3 = vorwärts(w1, b1, w2, b2, w3, b3, features_ev[i])

    if np.array_equal(a_to_output(a3), label_zu_vektor(labels_ev[i])):
        treffer += 1

print(treffer/200)





