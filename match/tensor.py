"""
- single list
- save shape and strides (property)
    + compute strides based on shape
- multiple dimensions
- row-major layout (not optional)
- methods to resize and reshape?
- indexing methods

- storage class as member of tensor class

"""

#%%
def compute_strides(shape):
    """
    docstring
    """
    data_size = 1
    strides = [0] * len(shape)
    for i in range(len(shape), 0, -1):
        strides[i - 1] = data_size
        data_size = strides[i - 1] * shape[i - 1]
    return data_size, strides


shape = (5, 7, 3)
data_size, strides = compute_strides(shape)
print(data_size, strides)
print(strides)

# %%
