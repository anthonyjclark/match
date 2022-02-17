
Some miscellaneous notes.

- storage
- matrix instead of tensor
- nn (layers and loss)
- create optim package
- binary mnist data in repo
- inspect.getmembers
- inspect.isfunction

```python
def stable_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)
```


