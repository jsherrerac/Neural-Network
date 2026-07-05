import numpy as np

b=np.random.randn()
w=np.random.randn(3)
def neuron(x)->float:
    """Returns the value of the neuron"""    
    dot = np.dot(w , x) + b
    return sigmoid(dot) 

def sigmoid(x):
    """Mathematical functions that returns a value (float) between 0 and 1"""
    return 1/(1+np.exp(-x))