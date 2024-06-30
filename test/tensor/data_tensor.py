import pytest
import torch
from macrograd.engine import Tensor
import numpy as np


@pytest.fixture(params=[
    ([1, 2, 3, 4], 1),
    ([1, 2, 3, 4], slice(3)),
    ([[1, 2, 3, 4], [5, 6, 7, 8]], slice(3)),
    ([[1, 2, 3, 4], [5, 6, 7, 8]], (slice(None), slice(2))),
    ([1, 2, 3, 4, 5, 6], slice(-1, -4, -1)),
    ([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], (2, slice(1, 3))),
    ([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], (slice(None, None, 2), slice(None, None, 2))),
    ([1, 2, 3, 4], slice(2, 2)),
    ([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], (slice(None), slice(1, 3))),
    ([[[1, 2, 3, 4]]], 0),
    ([[[1, 2, 3, 4], [5, 6, 7, 8]]], (0, slice(1))),
    ([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], (0, 1, 0)),
    ([[[1, 2, 3, 4, 5, 6]]], (0, 0, slice(0, 6, 2))),
    ([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]], (0, slice(-1, -3, -1))),
    ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], (1, 0, slice(None))),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], (slice(None), slice(None), 1)),
    ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]], (slice(1, None), 0, slice(None, None, 2))),
    ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]], (slice(1, None), 0, slice(None, None, 2))),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], (slice(None), slice(None), 1)),
    ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]], (slice(1, None), 0, slice(None, None, 2))),
    ([[[1, 2, 3, 4], [5, 6, 7, 8]]], (0, slice(2, 2))),
    ([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]], (slice(None), 1, slice(None, None, -2)))
])
def slicing_data(request):
    return request.param


@pytest.fixture(params=[
    # Existing tuple shape cases
    ([1, 2, 3, 4], (4,)),
    ([[1, 2], [3, 4], [5, 6]], (3, 2)),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], (2, 2, 2)),
    ([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], (1, 2, 2, 2)),
    ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], (2, 2, 3)),
    ([[[[1], [2]], [[3], [4]]], [[[5], [6]], [[7], [8]]]], (2, 2, 2, 1)),
    ([[[[[1]]]]], (1, 1, 1, 1, 1)),
    
    # New cases with integer shapes
    ([42], 1),
    ([1, 2, 3, 4, 5], 5),
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10),
    (list(range(100)), 100),
    ([3.14], 1),
    
    # Empty array
    ([], 0),
    
    # Large integer shape
    (list(range(1000)), 1000),
    
    # Additional tuple shape cases for variety
    ([[1], [2], [3], [4], [5]], (5, 1)),
    ([[[1, 2, 3]]], (1, 1, 3)),
    ([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], (4, 3)),
    ([[[1], [2], [3]], [[4], [5], [6]]], (2, 3, 1)),
    ([[[[1, 2, 3, 4]]]], (1, 1, 1, 4)),
])
def shape_data(request):
    return request.param


def create_macrograd_tensor(data):
    return Tensor(data)


def create_torch_tensor(data):
    return torch.tensor(data)


def create_numpy_array(data):
    return np.array(data)


def handle_initialization(data, lib):
    if lib == 'torch':
        macrograd, other = create_macrograd_tensor(data), create_torch_tensor(data)
    elif lib == 'numpy':
        macrograd, other = create_macrograd_tensor(data), create_numpy_array(data)
    else:
        raise ValueError(f"unsupported lib (got {lib}), must be one of ['torch', 'numpy']")
    
    return macrograd, other