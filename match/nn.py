from __future__ import annotations
import match

from match import Matrix

from math import sqrt


class Module:
    def __call__(self, *args) -> Matrix:
        return self.forward(*args)

    def forward(self, x) -> Matrix:
        raise NotImplementedError

    def parameters(self) -> list[Matrix]:
        params = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Matrix):
                params.append(attr)
        return params

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.grad.zeros_()


class Linear(Module):
    """
    y = x A^T + b
    """

    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        k = 1 / in_features
        self.A = match.randn(out_features, in_features) * sqrt(k)
        self.b = match.randn(out_features, 1) * sqrt(k)

    def forward(self, x: Matrix) -> Matrix:
        return x @ self.A.T + self.b.T

    def __repr__(self) -> str:
        return f"A: {self.A}\nb: {self.b}"


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
    """ """

    def forward(self, prediction: Matrix, target: Matrix) -> Matrix:
        return ((target - prediction) ** 2).mean()


# Conv2d, MaxPool2d, AdaptiveMaxPool2d, Flatten
# BatchNorm2d
# Dropout
# RNN, LSTM
# Embedding
# Transformer
# MSELoss, BCELoss, CrossEntropyLoss
