import torch
from mingrad.engine import Variable

def test_sanity_check():
    """
    Test the sanity check for forward and backward propagation.

    This function tests the computational graph using the `Variable` class 
    from micrograd and compares the results with PyTorch's autograd.
    """
    # Using micrograd
    x = Variable(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    # Using PyTorch
    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # Forward pass comparison
    assert ymg.data == ypt.data.item()
    # Backward pass comparison
    assert xmg.grad == xpt.grad.item()

def test_more_ops():
    """
    Test more complex operations for forward and backward propagation.

    This function tests the computational graph with more complex operations 
    using the `Variable` class from micrograd and compares the results with PyTorch's autograd.
    """
    # Using micrograd
    a = Variable(-4.0)
    b = Variable(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    # Using PyTorch
    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # Forward pass comparison
    assert abs(gmg.data - gpt.data.item()) < tol
    # Backward pass comparison
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol
