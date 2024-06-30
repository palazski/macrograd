from data_tensor import handle_initialization, slicing_data, shape_data
import pytest

@pytest.mark.parametrize("lib", ["torch", "numpy"])
def test_slicing(slicing_data, lib):
    data, slicing = slicing_data
    
    macrograd, other = handle_initialization(data, lib)

    result_macrograd = macrograd[slicing].tolist()
    result_other = other[slicing].tolist()

    assert result_macrograd == result_other


@pytest.mark.parametrize("lib", ["torch", "numpy"])
def test_shape(shape_data, lib):
    data, shape = shape_data

    macrograd, other = handle_initialization(data, lib)

    result_macrograd = list(macrograd.shape)
    result_other = list(other.shape)

    assert result_macrograd == result_other
