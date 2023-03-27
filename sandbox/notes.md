
# Some miscellaneous notes.

- https://karpathy.ai/zero-to-hero.html and https://github.com/karpathy/ng-video-lecture
- https://github.com/karpathy/nanoGPT
- https://github.com/karpathy/microGPT
- https://github.com/karpathy/micrograd
- storage
- matrix instead of tensor
- nn (layers and loss)
- create optim package
- binary mnist data in repo
- inspect.getmembers
- inspect.isfunction


# Broadcasting Rules

For `a + b`:

1. Pad the smaller (fewer dimensions) tensor with 1s on the left so that they have the same number of dimensions.
2. Match all dimensions by taking dimensions of size 1 and repeating them to match the larger value.
3. Error if the dimensions do not match.

# Function

```python
def stable_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)
```
