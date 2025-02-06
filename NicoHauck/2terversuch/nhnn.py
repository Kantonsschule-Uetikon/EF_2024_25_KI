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

        def rueckwaerts(self, dEdY, lern_rate):
            dEdw = np.outer(dEdY.reshape(-1, 1), self.input.reshape(-1, 1).T) #problem mit shape jetzt nicht mehr wegen outer
            dEdb = dEdY
            dEdX = (self.weights.T).dot(dEdY)

            self.weights = self.weights - lern_rate * dEdw
            self.biases = self.biases - lern_rate * dEdb
            return dEdX
            

    class Sigmoid(Layer):
        def __init__(self):
            pass
        
        def sigmoid(self, X):
            return 1 / (1 + np.exp(-X))
        
        def vorwaerts(self, X):
            self.input = X
            return self.sigmoid(X)

        def rueckwaerts(self, dEdY, lern_rate):
            dEdX = np.multiply(dEdY, self.sigmoid(self.input)*(1 - self.sigmoid(self.input)))
            return dEdX
        
    class Softmax(Layer):
        def __init__(self):
            pass
        

        def vorwaerts(self, X):
            self.input = X
            return np.exp(X) / np.sum(np.exp(X))
    
        def rueckwaerts(self, dEdY):
            #ableitung zu schwierig
            pass

    def train(self, epochen, lern_rate, X, Y):
        for i in range (epochen):
            for x, y in zip(X, Y):
                    output = x
                    for layer in self.layers:
                        output = layer.vorwaerts(output)
                        
                    error = y - output

                    for layer in reversed(self.layers):
                        error = layer.rueckwaerts(error, lern_rate)
        pass

    def evaluate(self):
        pass

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.vorwaerts(output)
        return output

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])

nn = nhnn([
    nhnn.Dense(2, 3),
    nhnn.Sigmoid(),
    nhnn.Dense(3, 1),
    nhnn.Sigmoid()
])

nn.train(10000, 0.001, X, Y)
print(nn.predict(np.array([0,0])))
print(nn.predict(np.array([0,1])))
print(nn.predict(np.array([1,0])))
print(nn.predict(np.array([1,1])))
print(nn.layers[0].weights)
print(nn.layers[2].weights)