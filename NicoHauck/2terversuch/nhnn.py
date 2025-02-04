import numpy as np

class nhnn():
    def __init__(self, layers):
        self.layers = layers
    
    class Layer():
        def __init__(self):
            pass
    
    class Dense(Layer):
        def __init__(self, input_neurons, output_neurons):
            self.input_neurons = input_neurons
            self.output_neurons = output_neurons
            self.weights = np.random.rand(output_neurons, input_neurons) - 0.5
            self.biases = np.random.rand(output_neurons) - 0.5

        def vorwaerts(self, X):
            self.input = X
            return self.weights.dot(X) + self.biases

        def rueckwaerts(self, dEdY):
            dEdw = dEdY.dot(self.input.T)
            dEdb = dEdY
            dEdX = (self.weights.T).dot(dEdY)

            self.weights -= dEdw
            self.biases -= dEdb
            return dEdX
            

    class Activation(Layer):
        def __init__(self):
            pass

    class Sigmoid(Activation):
        def __init__(self):
            pass

        def vorwaerts(self, X):
            return 1 / (1 + np.exp(-X))

        def rueckwaerts(self):
            pass
        
    class Softmax(Activation):
        def __init__(self):
            pass

        def vorwaerts(self, X):
            return np.exp(X) / np.sum(np.exp(X))
    
        def rueckwaerts(self):
            pass

    def train(self):
        pass

    def evaluate(self):
        pass



layer1 = nhnn.Dense(5, 2)
print(layer1.weights)
print(layer1.biases)
print(layer1.vorwaerts(np.array([0,1,2,3,4])))

layer2 = nhnn.Sigmoid()
print(layer2.vorwaerts(np.array([1,0,9,-10])))

layer2 = nhnn.Softmax()
print(layer2.vorwaerts(np.array([1,2,3])))
