import pytest
import torch
from macrograd.engine import Tensor

@pytest.fixture(params=[
    ([1, 2, 3, 4], 1),  # e.g. (4,)[1]
    ([1, 2, 3, 4], slice(3)),  # e.g. (4,)[:3]
    ([[1, 2, 3, 4], [5, 6, 7, 8]], slice(3)),  # e.g. (2, 4)[:3]
    ([[1, 2, 3, 4], [5, 6, 7, 8]], (slice(None), slice(2))),  # e.g. (2, 4)[:, :2]
    ([1, 2, 3, 4, 5, 6], slice(-1, -4, -1)),  # Expect: [6, 5, 4]
    ([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], (2, slice(1, 3))),  # Expect: [10, 11]
    ([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], (slice(None, None, 2), slice(None, None, 2))),  # Expect: [[1, 3], [9, 11]]
    ([1, 2, 3, 4], slice(2, 2)),  # Expect: []
    ([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], (slice(None), slice(1, 3))),  # Expect: [[2, 3], [6, 7], [10, 11], [14, 15]]
    ([[[1, 2, 3, 4]]], 0),  # Expect: [[1, 2, 3, 4]]
    ([[[1, 2, 3, 4], [5, 6, 7, 8]]], (0, slice(1))),  # Expect: [[1, 2, 3, 4]]
    ([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], (0, 1, 0)),  # Expect: [5, 6]
    ([[[1, 2, 3, 4, 5, 6]]], (0, 0, slice(0, 6, 2))),  # Expect: [1, 3, 5]
    ([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]], (0, slice(-1, -3, -1))),  # Expect: [[5, 6, 7, 8], [1, 2, 3, 4]]
    ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], (1, 0, slice(None))),  # Expect: [7, 8, 9]
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], (slice(None), slice(None), 1)),  # Expect: [[2, 4], [6, 8], [10, 12]]
    ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]], (slice(1, None), 0, slice(None, None, 2))),  # Expect: [[7, 9], [13, 15]]
    ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]], (slice(1, None), 0, slice(None, None, 2))),  # Expect: [[7, 9], [13, 15]]
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], (slice(None), slice(None), 1)),  # Expect: [[2, 4], [6, 8], [10, 12]]
    ([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[13, 14, 15], [16, 17, 18]]], (slice(1, None), 0, slice(None, None, 2))),  # Expect: [[7, 9], [13, 15]]
    ([[[1, 2, 3, 4], [5, 6, 7, 8]]], (0, slice(2, 2))),  # Expect: []
    ([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]], (slice(None), 1, slice(None, None, -2)))  # Expect: [[8, 6], [16, 14]]
])
def slicing_data(request):
    return request.param

def test_slicing_against_torch(slicing_data):
    data, slicing = slicing_data

    macrograd_tensor = Tensor(data)
    torch_tensor = torch.tensor(data)

    result_macrograd = macrograd_tensor[slicing].tolist()
    result_torch = torch_tensor[slicing].tolist()

    assert result_macrograd == result_torch
