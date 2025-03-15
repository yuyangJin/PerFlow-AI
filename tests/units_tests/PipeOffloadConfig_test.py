'''
test PipeOffloadConfig class
'''
import pytest

from perflowai.parallel.pipeline_parallel import PipeOffloadConfig

def test_PipeOffloadConfig():
    with pytest.raises(AssertionError):
        poc = PipeOffloadConfig(offload_ratio = 2)
    
    poc = PipeOffloadConfig(offload_ratio = 0.5)
    assert poc.offload_ratio == 0.5

    poc = PipeOffloadConfig(offload_ratio = [0.5, 0.5])
    assert poc.offload_ratio == [0.5, 0.5]

    poc = PipeOffloadConfig(offload_ratio = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    assert poc.offload_ratio ==  [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]