import random
from micrograd.engine import Value

class Module:
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def SGDoptimize(self, lr=0.1):
        for p in self.parameters():
            p.data += -lr * p.grad

    def parameters(self):
        return []

class Neuron(Module):

    #Create a neuron with 'nin' inputs
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __repr__(self):
        return f'Neuron that takes in {len(self.w)} inputs'

    #Forward pass through the neuron with input vector x of size 'nin'
    def __call__(self, x):
        #w * x + b -> raw activation
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)

        #Apply activation function
        out = act.relu() if self.nonlin else act
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer(Module):

    #Create a layer of neurons
    #There are nin' inputs going into each neuron
    #There are 'nout' neurons in the layer (number of outputs)
    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def __repr__(self):
        return f'Layer of {len(self.neurons)} neurons'
    
    #Forward pass through the layer with input vector x of size 'nin'
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP(Module):
    
    #Create a multi-layered perception
    #The MLP takes in 'nin' inputs
    #Layer i has nouts[i] neurons
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i!=len(nouts)-1)) for i in range(len(nouts))]

    def __repr__(self):
        return f'MLP with {len(self.layers)} layers'

    #Forward pass through the MLP with inout vector x of size 'nin'
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]