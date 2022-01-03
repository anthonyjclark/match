# match
A simple PyTorch-like neural network library for education

- storage
- matrix instead of tensor
- nn (layers and loss)
- optim
- binary mnist data in repo

```python
def stable_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)
```

# Exercises

- Add LeakyReLU
- Add MAE (and CE?)

# Testing

```bash
python -m unittest test
```
# Resources

- [Build Your own Deep Learning Framework - A Hands-on Introduction to Automatic Differentiation - Part 2](https://mostafa-samir.github.io/auto-diff-pt2/ "Build Your own Deep Learning Framework - A Hands-on Introduction to Automatic Differentiation - Part 2")
- [How we wrote xtensor 1/N: N-Dimensional Containers | by Johan Mabille | Medium](https://johan-mabille.medium.com/how-we-wrote-xtensor-1-n-n-dimensional-containers-f79f9f4966a7 "How we wrote xtensor 1/N: N-Dimensional Containers | by Johan Mabille | Medium")
- [Hands-on-Intro-to-Auto-Diff/Chain-Rule-on-Matrix-Multiplication.pdf at master · Mostafa-Samir/Hands-on-Intro-to-Auto-Diff](https://github.com/Mostafa-Samir/Hands-on-Intro-to-Auto-Diff/blob/master/Chain-Rule-on-Matrix-Multiplication.pdf "Hands-on-Intro-to-Auto-Diff/Chain-Rule-on-Matrix-Multiplication.pdf at master · Mostafa-Samir/Hands-on-Intro-to-Auto-Diff")
- [karpathy/micrograd: A tiny scalar-valued autograd engine and a neural net library on top of it with PyTorch-like API](https://github.com/karpathy/micrograd "karpathy/micrograd: A tiny scalar-valued autograd engine and a neural net library on top of it with PyTorch-like API")
- [The magic behind autodiff | Tutorials on automatic differentiation and JAX](https://sscardapane.github.io/learn-autodiff/ "The magic behind autodiff | Tutorials on automatic differentiation and JAX")
