import math

class Variable:
    """
    Represents a variable that stores a single scalar value and its gradient.
    """

    def __init__(self, data, _children=(), _op=''):
        """
        Initialize Variable object.

        Args:
            data: The scalar value to store.
            _children: Internal parameter for autograd graph construction.
            _op: The operation that produced this node.
        """
        self.data = data
        self.grad = 0
        # Internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # The op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Variable(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Variable(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        out = Variable(1 / (1 + math.exp(-self.data)), (self,), 'sigmoid')

        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward

        return out

    def gelu(self):
        def gelu_fn(x):
            return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
        
        out = Variable(gelu_fn(self.data), (self,), 'gelu')

        def _backward():
            x = self.data
            tanh_term = math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3))
            derivative = 0.5 * (1 + tanh_term) + (0.5 * x * (1 - tanh_term**2) * (math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * x**2)))
            self.grad += derivative * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        exp_2x = math.exp(2 * self.data)
        out = Variable((exp_2x - 1) / (exp_2x + 1), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward

        return out

    def leaky_relu(self, alpha=0.01):
        out = Variable(self.data if self.data > 0 else alpha * self.data, (self,), 'leaky_relu')

        def _backward():
            self.grad += (1 if self.data > 0 else alpha) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        out = Variable(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """
        Perform backpropagation to compute the gradients of all variables in the computational graph.
        """
        # Topological order all of the children in the graph
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)

        # Go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __repr__(self):
        return f"Variable(data={self.data}, grad={self.grad})"
