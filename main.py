import numpy as np
def sigmoid(x):
    """Mathematical functions that returns a value (float) between 0 and 1"""
    return 1/(1+np.exp(-x))
class Neuron:
    def __init__(self, n_inputs):
        self.w= np.random.randn(n_inputs)
        self.b=np.random.randn()
    
    def forward(self, x):
        return sigmoid(np.dot(self.w, x)+self.b)
    
class Network:
    def __init__(self, sizes):
        self.sizes= sizes
        self.weights= [np.random.randn(15,784), np.random.randn(10,15)]
        self.biases= [np.random.randn(15),np.random.randn(10)]
        
    def feedforward(self, a):
        for w,b in zip(self.weights, self.biases):
            a=sigmoid(w@a + b)
        return a
net = Network([784, 15, 10])
salida = net.feedforward(np.random.rand(784))
print(salida.shape)