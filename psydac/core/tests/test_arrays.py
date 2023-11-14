import numpy as np
import pytest

from psydac.core.arrays import matmul, sum_vec, min_vec, max_vec

@pytest.mark.parametrize('seed', [123, 6, 49573])
def test_arrays(seed):
    
    np.random.seed(seed)
    
    a = np.random.rand(30, 20)
    b = np.random.rand(20, 40)
    c = np.zeros((30, 40))
    
    matmul(a, b, c)
    assert np.allclose(np.matmul(a, b), c)
    
    v = np.random.rand(32)
    assert np.isclose(np.sum(v), sum_vec(v))
    assert np.isclose(np.min(v), min_vec(v))
    assert np.isclose(np.max(v), max_vec(v))

    
if __name__ == '__main__':
    test_arrays(459)