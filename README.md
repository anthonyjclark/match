# Match

A pure-Python, PyTorch-like automatic differentiation library for education. Here is the directory structure of the repository.

~~~text
.
├── match               # The Match library
│  ├── __init__.py      # - contains default import statements
│  ├── list2d.py        # - a storage class for matrix data
│  ├── matrix.py        # - the matrix class (including autodiff)
│  └── nn.py            # - higher-level neural network functionality
├── demo_linear.ipynb   # A linear model demo (Jupyter)
├── demo_linear.py      # A linear model demo (script)
├── test.py             # Unit-tests for correctness
├── LICENSE             # MIT License
└── README.md           # This document
~~~

https://user-images.githubusercontent.com/4173647/154419094-5787e3a5-0e69-4d89-9ed7-e3ee507f1f32.mp4

# Demos

Although **Match** does not have any dependencies, the demos do. Demos import [matplotlib](https://matplotlib.org/), but you can skip (or comment) plotting cells (code) and not miss out on much. Demos come in two flavors: Jupyter Notebooks and Python scripts. These files are synced using [Jupytext](https://jupytext.readthedocs.io/en/latest/ "Jupyter Notebooks as Markdown Documents, Julia, Python or R Scripts — Jupytext documentation").

# Implementation of Automatic Differentiation

Here are the highlights of the implementation:

- `list2d.py` contains an implementation of several matrix operations (e.g., element-wise arithmetic, matrix multiplication, etc.)
- `matrix.py` relies on `list2d.py` and adds automatic differentiation functionality (it was important to decouple the underlying matrix operations in `list2d.py` from the usage of `Matrix` objects; this made it easier to include matrix operations in the gradient functions without running into a recursive loop)
- `nn.py` adds common neural network functionality on top of `matrix.py`

Here is an example showing how gradients are computed when the sigmoid activation function is involved.

~~~python
def sigmoid(self) -> Matrix:
    """Element-wise sigmoid."""
    result = Matrix(self.data.sigmoid(), children=(self,))

    def _gradient() -> None:
        self.grad += result.data * (1 - result.data) * result.grad

    result._gradient = _gradient
    return result
~~~

The following occurs when `sigmoid` is called on an existing `Matrix`:

1. A new `Matrix` (called `result`) is constructed
  + Elements in the new `Matrix` are computed by taking the `sigmoid` of each value
  + The new matrix is the same shape as the original matrix (`self`)
  + The current matrix is passed as a child into the new matrix
2. A gradient function (called `_gradient`) is created with the correct computations
  + We cannot compute the gradient of `self` until we have the gradient of `result`, which is not computed until its parents are computed
  + The gradient of sigmoid is: σ(z) * (1 - σ(z))
  + The additional term `result.grad` is the backward component

# Testing

The library has a limited number of tests in the file `test.py` found in the root directory. Unit tests require the PyTorch library. They should be executed with:

~~~bash
python -m unittest test
~~~

# Sandbox

The `sandbox` directory can be ignored. That is where I am putting ideas for future updates.

# Resources

- [Build Your own Deep Learning Framework - A Hands-on Introduction to Automatic Differentiation - Part 2](https://mostafa-samir.github.io/auto-diff-pt2/ "Build Your own Deep Learning Framework - A Hands-on Introduction to Automatic Differentiation - Part 2")
- [How we wrote xtensor 1/N: N-Dimensional Containers | by Johan Mabille](https://johan-mabille.medium.com/how-we-wrote-xtensor-1-n-n-dimensional-containers-f79f9f4966a7 "How we wrote xtensor 1/N: N-Dimensional Containers | by Johan Mabille")
- [Chain Rule on Matrix Multiplication](https://github.com/Mostafa-Samir/Hands-on-Intro-to-Auto-Diff/blob/master/Chain-Rule-on-Matrix-Multiplication.pdf "Hands-on-Intro-to-Auto-Diff/Chain-Rule-on-Matrix-Multiplication.pdf at master · Mostafa-Samir/Hands-on-Intro-to-Auto-Diff")
- [A tiny scalar-valued autograd engine and a neural net library on top of it with PyTorch-like API](https://github.com/karpathy/micrograd "karpathy/micrograd: A tiny scalar-valued autograd engine and a neural net library on top of it with PyTorch-like API")
- [The magic behind autodiff | Tutorials on automatic differentiation and JAX](https://sscardapane.github.io/learn-autodiff/ "The magic behind autodiff | Tutorials on automatic differentiation and JAX")
- [Google Colab: Coding a neural net](https://colab.research.google.com/drive/1HS3qbHArkqFlImT2KnF5pcMCz7ueHNvY?usp=sharing#scrollTo=RWqEaOWqNbwV)
- [Example matrix gradients](https://github.com/Mostafa-Samir/Hands-on-Intro-to-Auto-Diff/blob/master/autodiff/grads.py)
