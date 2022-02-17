"""
TODO(AJC): things for me to add in the future.

- Conv2d, MaxPool2d, AdaptiveMaxPool2d, Flatten
- BatchNorm2d
- Dropout
- RNN, LSTM
- Embeddings
- Transformer
- BCELoss, CrossEntropyLoss
"""


from __future__ import annotations

from math import sqrt

import match
from match import Matrix


class Module:
    """Base class for all neural network modules.

    All custom models should subclass this class. Modules can also
    contain other Modules, allowing to nest them in a tree structure.
    You can assign the submodules as regular attributes:


    """

    def __call__(self, *args) -> Matrix:
        """Enable calling the module like a function."""
        return self.forward(*args)

    def forward(self) -> Matrix:
        """Forward must be implemented by the subclas."""
        raise NotImplementedError("Implement in the subclass.")

    def parameters(self) -> list[Matrix]:
        """Collect all parameters by searching attributes for Matrix objects."""
        modules_with_parameters = (Linear,)
        params = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, modules_with_parameters):
                params.append(attr.W)
                params.append(attr.b)
        return params

    def zero_grad(self) -> None:
        """Set gradients for all parameters to zero."""
        for param in self.parameters():
            param.grad.zeros_()


class Linear(Module):
    """
    y = x W^T + b
    """

    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        # Kaiming He initialization
        self.W = match.randn(out_features, in_features) * sqrt((2 / out_features) / 3)
        self.b = match.randn(out_features, 1) * sqrt((2 / out_features) / 3)

    def forward(self, x: Matrix) -> Matrix:
        return x @ self.W.T + self.b.T

    def __repr__(self) -> str:
        return f"A: {self.W}\nb: {self.b}"


class ReLU(Module):
    """
    ReLU(x) = max(0, x)
    """

    def forward(self, x: Matrix) -> Matrix:
        return x.relu()


class Sigmoid(Module):
    """
    Sigmoid(x) = 1 / (1 + e^(-x))
    """

    def forward(self, x: Matrix) -> Matrix:
        return x.sigmoid()


class MSELoss(Module):
    """
    loss = (1/N) * Σ (yhati - yi)^2
    """

    def forward(self, prediction: Matrix, target: Matrix) -> Matrix:
        return ((target - prediction) ** 2).mean()
