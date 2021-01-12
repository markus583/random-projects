import numpy as np
from torch import tensor
import torch


def matmul(a, b):
    c = np.zeros(shape=(a.shape[0], b.shape[1]))
    counter = 0  # counter == i * j* k
    for k in range(b.shape[1]):  # n(columns) of b
        for i in range(a.shape[0]):  # n(rows) of a
            for j in range(a.shape[1]):  # n(columns) of a
                c[i, k] += a[i, j] * b[j, k]
                counter += 1
    return c, counter


def fast_matmul(a, b):
    c = np.zeros(shape=(a.shape[0], b.shape[1]))
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            # Any trailing ",:" can be removed
            c[i, j] = (a[i, :] * b[:, j]).sum()
    return c


def nested_matmul(a, b):
    c = [[sum(a * b for a, b in zip(A_row, B_col))
          for B_col in zip(*b)]
         for A_row in a]
    return np.array(c)


def einsum_matmul(a, b):
    c = np.einsum('ij,jk->ik', a, b)
    return c


def numpy_matmul(a, b):
    return a @ b


def broadcasting_matmul(a, b):
    a = tensor(a)
    b = tensor(b)
    c = torch.zeros(a.shape[0], b.shape[1])
    for i in range(a.shape[0]):
        c[i] = (a[i].unsqueeze(-1) * b).sum(dim=0)
    return c


if __name__ == '__main__':
    a_list = [[-3], [5], [6], [1]]
    a = np.array(a_list)
    print(a)

    b_list = [[0, 4, -4]]
    b = np.array(b_list)
    print(b)

    # check if shape is correct
    assert a.shape[1] == b.shape[0]

    true, counter = matmul(a, b)
    print(true, counter)

    nested = nested_matmul(a, b)
    print(nested)

    ein = einsum_matmul(a, b)
    print(ein)

    fast = fast_matmul(a, b)
    print(fast)

    numpy_mm = numpy_matmul(a, b)
    print(numpy_mm)

    broadcasted_mm = broadcasting_matmul(a, b)
    print(broadcasted_mm)
