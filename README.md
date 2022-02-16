# match

A pure-Python, PyTorch-like automatic differentiation library for education. Here is the directory structure of the repository.

~~~bash
.
├── match               # The match library
│  ├── __init__.py      # - contains default import statements
│  ├── list2d.py        # - a storage class for matrix data
│  ├── matrix.py        # - the matrix class (including autodiff)
│  └── nn.py            # - higher-level neural network functionality
├── demo_linear.ipynb   # A linear model demo
├── test.py             # Unit-tests for correctness
├── LICENSE             # MIT License
└── README.md           # This document
~~~

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

