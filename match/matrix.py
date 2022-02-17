# TODO:
# - commenting (asserts, classes, etc.)
# - document function name conventions (_ vs not)
# - update parameters without updating gradients
# - abs, __abs__, sin, cos, others?

from __future__ import annotations

from .list2d import List2D


def full(nrow: int, ncol: int, val: float, children: tuple = ()) -> Matrix:
    """Create a matrix of given size using the fill value.

    Args:
        nrow (int): number of rows
        ncol (int): number of columns
        val (float): fill value
        children (tuple, optional): for backpropagation. Defaults to ().

    Returns:
        Matrix: a nrow by ncol matrix filled with val
    """
    return Matrix(List2D(nrow, ncol, val), children)


def randn(nrow: int, ncol: int, children: tuple = ()) -> Matrix:
    """Create a matrix of the given size filled with normally distributed random values.

    Args:
        nrow (int): number of rows
        ncol (int): number of columns
        children (tuple, optional): for backpropagation. Defaults to ().

    Returns:
        Matrix: a nrow by ncol matrix filled with random values
    """
    return Matrix(List2D.randn(nrow, ncol), children)


def zeros(nrow: int, ncol: int, children: tuple = ()) -> Matrix:
    """Create a matrix of given size filled with zeros.

    Args:
        nrow (int): number of rows
        ncol (int): number of columns
        children (tuple, optional): for backpropagation. Defaults to ().

    Returns:
        Matrix: a nrow by ncol matrix filled with val
    """
    return full(nrow, ncol, 0.0, children)


def ones(nrow: int, ncol: int, children: tuple = ()) -> Matrix:
    """Create a matrix of given size filled with ones.

    Args:
        nrow (int): number of rows
        ncol (int): number of columns
        children (tuple, optional): for backpropagation. Defaults to ().

    Returns:
        Matrix: a nrow by ncol matrix filled with val
    """
    return full(nrow, ncol, 1.0, children)


def mat(values: list[list[float]]) -> Matrix:
    return Matrix(List2D.fromData(values))


class Matrix(object):
    def __init__(self, data: List2D, children: tuple = ()) -> None:
        super().__init__()
        self.nrow = data.nrow
        self.ncol = data.ncol
        self.shape = (self.nrow, self.ncol)

        self.data = data
        self.grad = List2D(self.nrow, self.ncol, 0.0)

        # Backpropagation compute graph
        self._gradient = lambda: None
        self._children = set(children)

    def __repr__(self) -> str:
        return self.data.__repr__()

    def __str__(self) -> str:
        return self.__repr__()

    def backward(self) -> None:
        """Compute all gradients."""

        sorted_nodes: list[Matrix] = []
        visited: set[Matrix] = set()

        # Sort all elements in the compute graph using a topological ordering (DFS)
        def topological_sort(node: Matrix) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    topological_sort(child)
                sorted_nodes.append(node)

        topological_sort(self)

        self.grad.ones_()

        # Update gradients from output to input
        for node in reversed(sorted_nodes):
            node._gradient()

    @property
    def T(self) -> Matrix:
        """Return a transposed version of this matrix."""
        result = Matrix(self.data.T, children=(self,))

        def _gradient() -> None:
            self.grad += result.grad.T

        result._gradient = _gradient
        return result

    def sum(self) -> Matrix:
        result = Matrix(List2D(1, 1, self.data.sum()), children=(self,))

        def _gradient() -> None:
            self.grad += List2D(self.nrow, self.ncol, result.grad.data[0][0])

        result._gradient = _gradient
        return result

    def mean(self) -> Matrix:
        result = Matrix(List2D(1, 1, self.data.mean()), children=(self,))

        def _gradient() -> None:
            n = self.nrow * self.ncol
            self.grad += List2D(self.nrow, self.ncol, result.grad.data[0][0] / n)

        result._gradient = _gradient
        return result

    def log(self) -> Matrix:
        raise NotImplementedError

    def max(self) -> Matrix:
        raise NotImplementedError

    def relu(self) -> Matrix:
        """Element-wise rectified linear unit (ReLU)."""
        result = Matrix(self.data.relu(), children=(self,))

        def _gradient() -> None:
            self.grad += (result.data > 0) * result.grad

        result._gradient = _gradient
        return result

    def sigmoid(self) -> Matrix:
        """Element-wise sigmoid."""
        result = Matrix(self.data.sigmoid(), children=(self,))

        def _gradient() -> None:
            self.grad += result.data * (1 - result.data) * result.grad

        result._gradient = _gradient
        return result

    def __add__(self, rhs: float | int | Matrix) -> Matrix:
        """Element-wise addition."""
        assert isinstance(rhs, (float, int, Matrix)), f"Wrong type: {type(rhs)}"

        rhsvals = rhs.data if isinstance(rhs, Matrix) else rhs
        children = (self, rhs) if isinstance(rhs, Matrix) else (self,)
        result = Matrix(self.data + rhsvals, children=children)

        def _gradient() -> None:
            self.grad += result.grad.unbroadcast(*self.shape)
            if isinstance(rhs, Matrix):
                rhs.grad += result.grad.unbroadcast(*rhs.shape)

        result._gradient = _gradient
        return result

    def __mul__(self, rhs: float | int | Matrix) -> Matrix:
        """Element-wise multiplication."""
        assert isinstance(rhs, (float, int, Matrix)), f"Wrong type: {type(rhs)}"

        rhsvals = rhs.data if isinstance(rhs, Matrix) else rhs
        children = (self, rhs) if isinstance(rhs, Matrix) else (self,)
        result = Matrix(self.data * rhsvals, children=children)

        def _gradient() -> None:
            self.grad += (rhsvals * result.grad).unbroadcast(*self.shape)
            if isinstance(rhs, Matrix):
                rhs.grad += (self.data * result.grad).unbroadcast(*rhs.shape)

        result._gradient = _gradient
        return result

    def __pow__(self, rhs: float | int) -> Matrix:
        """Element-wise exponentiation: self^rhs."""
        assert isinstance(rhs, (float, int)), f"Wrong type: {type(rhs)}"

        result = Matrix(self.data**rhs, children=(self,))

        def _gradient() -> None:
            # rhsvals will be a number (not matrix)
            g = rhs * self.data ** (rhs - 1) * result.grad
            self.grad += g.unbroadcast(*self.shape)

        result._gradient = _gradient
        return result

    def __matmul__(self, rhs: Matrix) -> Matrix:
        """Matrix multiplication: self @ rhs."""
        assert isinstance(rhs, Matrix), f"Wrong type: {type(rhs)}"
        assert self.ncol == rhs.nrow, f"Wrong shapes: {self.shape} and {rhs.shape}"

        result = Matrix(self.data @ rhs.data, children=(self, rhs))

        def _gradient() -> None:
            self.grad += result.grad @ rhs.data.T
            rhs.grad += self.data.T @ result.grad

        result._gradient = _gradient
        return result

    def __radd__(self, lhs: float | int) -> Matrix:
        """Element-wise addition is commutative: lhs + self."""
        return self + lhs

    def __sub__(self, rhs: float | int | Matrix) -> Matrix:
        """Element-wise subtraction: self - rhs."""
        return -rhs + self

    def __rsub__(self, lhs: float | int) -> Matrix:
        """Self as RHS in element-wise subtraction: lhs - self."""
        return -self + lhs

    def __rmul__(self, lhs: float | int) -> Matrix:
        """Self as RHS in element-wise multiplication: lhs * self."""
        return self * lhs

    def __truediv__(self, rhs: float | int) -> Matrix:
        """Element-wise division: self / rhs."""
        return self * rhs**-1

    def __rtruediv__(self, lhs: float | int) -> Matrix:
        """Self as RHS in element-wise division: lhs / self."""
        return lhs * self**-1

    def __neg__(self) -> Matrix:
        """Element-wise unary negation: -self."""
        return self * -1
