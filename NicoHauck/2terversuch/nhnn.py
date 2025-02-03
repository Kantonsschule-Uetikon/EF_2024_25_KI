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
            self.weights = np.randn(output_neurons, input_neurons)

        def vorwaerts(self):
            pass
        
        def rueckwaerts(self):
            pass

    class Activation(Layer):
        def __init__(self, neurons):
            pass

    class Sigmoid(Activation):
        def __init__(self, neurons):
            pass

        def vorwaerts(self):
            pass
        def rueckwaerts(self):
            pass
        
    class Softmax(Activation):
        def __init__(self, neurons):
            pass

        def vorwaerts(self):
            pass
        def rueckwaerts(self):
            pass

    def train(self):
        pass

    def evaluate(self):
        pass