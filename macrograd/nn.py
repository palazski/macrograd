import random
from macrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Linear(Module):

    def __init__(self, features_in, features_out, bias=True, **kwargs):
        self.weight = [Value(0) for f in range(features_out)]
        self.bias = None
        if bias:
            self.bias = [Value(0) for f in range(features_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    


class ReLU(Module):
    def __call__(self, x):
        out = [Value(0 if x_i.data < 0 else x_i.data, (x_i,), 'ReLU') for x_i in x]
    

    def relu(self):
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out