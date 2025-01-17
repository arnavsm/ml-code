import torch
from torch import nn, optim
from torch.nn import functional as F


def describe(x: torch.Tensor):
    print(f"Type: {x.type()}") 
    print(f"Shape/size: {x.shape}") 
    print(f"Values: \n{x}")

def divider():
    print("\n")
    print("---------------------------------------------------------------")
    print("\n")