'''
test PipeRecomputeConfig class
'''

import pytest

from perflowai.parallel.pipeline_parallel import PipeRecomputeConfig

def test_PipeRecomputeConfig():

    # Invalid case 1: the mask value is more not 0 or 1 
    with pytest.raises(AssertionError):
        prc = PipeRecomputeConfig(recompute_mask = [0,1,2])
    with pytest.raises(AssertionError):
        prc = PipeRecomputeConfig(recompute_mask = [[0,1], [1,-1]])

    # Invalid case 2: the mask value is not int
    with pytest.raises(AssertionError):
        prc = PipeRecomputeConfig(recompute_mask = [0, [0,1]])
    with pytest.raises(AssertionError):
        prc = PipeRecomputeConfig(recompute_mask = [0,0.5])
    with pytest.raises(AssertionError):
        prc = PipeRecomputeConfig(recompute_mask = [[0,1], [1,1.0]])

    # Invalid case 3: the length of mask is not the same
    with pytest.raises(AssertionError):
        prc = PipeRecomputeConfig(recompute_mask = [[0,1], [1,0,1]])

    # Valid case
    prc = PipeRecomputeConfig(recompute_mask = [[0,1], [1,0]])

    assert prc.recompute_mask == [[0,1], [1,0]]