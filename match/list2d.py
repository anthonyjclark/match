from __future__ import annotations
from operator import add, mul, pow, gt
from math import exp
from random import gauss
from typing import Callable


def sigmoid(z):
    return (1 / (1 + exp(-z))) if z > 0 else (1 - 1 / (1 + exp(z)))


class List2D(object):
    def __init__(
        self, nrow: int, ncol: int, val: float | list[list[float]] = 0.0
    ) -> None:
        super().__init__()

        self.nrow = nrow
        self.ncol = ncol
        self.shape = (nrow, ncol)

        if isinstance(val, (float, int)):
            self.data = [[val] * ncol for _ in range(nrow)]
        elif (
            isinstance(val, list)
            and isinstance(val[0], list)
            and isinstance(val[0][0], (float, int))
        ):
            self.data = val
        else:
            raise TypeError("Cannot create List2D from", type(val))

    @staticmethod
    def fromData(vals: list[list[float]]) -> List2D:
        return List2D(len(vals), len(vals[0]), vals)

    @staticmethod
    def randn(nrow: int, ncol: int) -> List2D:
        data = [[gauss(0, 1) for _ in range(ncol)] for _ in range(nrow)]
        return List2D.fromData(data)

    def ones_(self) -> None:
        self.__set(1.0)

    def zeros_(self) -> None:
        self.__set(0.0)

    def sum(self) -> float:
        return sum(self.data[i][j] for j in range(self.ncol) for i in range(self.nrow))

    def mean(self) -> float:
        return self.sum() / (self.nrow * self.ncol)

    def relu(self) -> List2D:
        data = [
            [max(0.0, self.data[i][j]) for j in range(self.ncol)]
            for i in range(self.nrow)
        ]
        return List2D(*self.shape, data)

    def sigmoid(self) -> List2D:
        data = [
            [sigmoid(self.data[i][j]) for j in range(self.ncol)]
            for i in range(self.nrow)
        ]
        return List2D(*self.shape, data)

    def __repr__(self) -> str:
        return "\n".join(" ".join(f"{val: 2.4f}" for val in row) for row in self.data)

    def __str__(self) -> str:
        return self.__repr__()

    def broadcast(self, nrow: int, ncol: int) -> List2D:
        if self.nrow == nrow and self.ncol == ncol:
            return self
        elif self.nrow == 1 and self.ncol == 1:
            return List2D(nrow, ncol, self.data[0][0])
        elif self.nrow == 1:
            return List2D.fromData(self.data * nrow)
        else:  # self.ncol == 1
            data = [row * ncol for row in self.data]
            return List2D.fromData(data)

    def unbroadcast(self, nrow: int, ncol: int) -> List2D:
        if self.nrow == nrow and self.ncol == ncol:
            return self
        elif nrow == 1 and ncol == 1:
            return List2D(nrow, ncol, self.sum())
        elif nrow == 1:
            return List2D.fromData([list(map(sum, zip(*self.data)))])
        else:  # self.ncol == 1
            return List2D.fromData([[sum(row)] for row in self.data])

    @property
    def T(self) -> List2D:
        out = List2D(self.ncol, self.nrow)
        for i in range(out.nrow):
            for j in range(out.ncol):
                out.data[i][j] = self.data[j][i]
        return out

    def __set(self, val) -> None:
        self.data = [[val] * self.ncol for _ in range(self.nrow)]

    def __binary_op(self, op: Callable, rhs: float | int | List2D) -> List2D:

        if isinstance(rhs, (float, int)):
            rhs = List2D(*self.shape, rhs)

        if isinstance(rhs, List2D):
            if self.nrow != rhs.nrow and self.nrow != 1 and rhs.nrow != 1:
                raise TypeError(f"Wrong shapes: {self.shape} and {rhs.shape}")
            if self.ncol != rhs.ncol and self.ncol != 1 and rhs.ncol != 1:
                raise TypeError(f"Wrong shapes: {self.shape} and {rhs.shape}")
        else:
            raise TypeError(f"Wrong type: {type(rhs)}")

        nrow = max(self.nrow, rhs.nrow)
        ncol = max(self.ncol, rhs.ncol)

        out = List2D(nrow, ncol)

        lhs = self.broadcast(nrow, ncol)
        rhs = rhs.broadcast(nrow, ncol)

        for i in range(out.nrow):
            for j in range(out.ncol):
                out.data[i][j] = op(lhs.data[i][j], rhs.data[i][j])

        return out

    def __add__(self, rhs: float | int | List2D) -> List2D:
        """Element-wise addition: self + rhs."""
        return self.__binary_op(add, rhs)

    def __radd__(self, lhs: float | int | List2D) -> List2D:
        """Element-wise addition is commutative: lhs + self."""
        return self + lhs

    def __sub__(self, rhs: float | int | List2D) -> List2D:
        """Element-wise subtraction: self - rhs."""
        return -rhs + self

    def __rsub__(self, lhs: float | int | List2D) -> List2D:
        """Self as RHS in element-wise subtraction: lhs - self."""
        return -self + lhs

    def __mul__(self, rhs: float | int | List2D) -> List2D:
        """Element-wise multiplication: self * rhs."""
        return self.__binary_op(mul, rhs)

    def __rmul__(self, lhs: float | int | List2D) -> List2D:
        """Self as RHS in element-wise multiplication: lhs * self."""
        return self * lhs

    def __truediv__(self, rhs: float | int | List2D) -> List2D:
        """Element-wise division: self / rhs."""
        return self * rhs ** -1

    def __rtruediv__(self, lhs: float | int | List2D) -> List2D:
        """Self as RHS in element-wise division: lhs / self."""
        return lhs * self ** -1

    def __pow__(self, rhs: float | int) -> List2D:
        """Element-wise exponentiation: self ** rhs."""
        assert isinstance(rhs, (float, int)), "Exponent must be a number."
        exponent = List2D(self.nrow, self.ncol, rhs)
        return self.__binary_op(pow, exponent)

    def __neg__(self) -> List2D:
        """Element-wise unary negation: -self."""
        negative_ones = List2D(self.nrow, self.ncol, -1)
        return self * negative_ones

    def __matmul__(self, rhs: List2D) -> List2D:
        """Two-dimensional matrix multiplication."""
        assert self.ncol == rhs.nrow, "Mismatched shapes in matmul."

        out = List2D(self.nrow, rhs.ncol)

        for i in range(out.nrow):
            for j in range(out.ncol):
                for k in range(self.ncol):
                    out.data[i][j] += self.data[i][k] * rhs.data[k][j]

        return out

    def __gt__(self, rhs: float | int | List2D) -> List2D:
        """Element-wise comparison: self > rhs."""
        return self.__binary_op(gt, rhs)
