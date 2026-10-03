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
        self.weights= [np.random.randn(sizes[1],sizes[0]), np.random.randn(sizes[2],sizes[1])]
        self.biases= [np.random.randn(sizes[1]),np.random.randn(sizes[2])]
        
    def feedforward(self, a):
        for w,b in zip(self.weights, self.biases):
            a=sigmoid(w@a + b)
        return a