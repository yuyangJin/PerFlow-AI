'''
test PipeCostConfig class
'''
import pytest

from perflowai.parallel.pipeline_parallel import PipeCostConfig

def test_PipeCostConfig():

    # Default
    pcc = PipeCostConfig()

    assert pcc.fwd_time == 1
    assert pcc.bwd_time == 1
    assert pcc.wgt_time == 1
    assert pcc.comm_time == 1
    assert pcc.fwd_mem == 2.0
    assert pcc.bwd_mem == -2.0
    assert pcc.wgt_mem == 0.0

    # Invalid 
    with pytest.raises(AssertionError):
        pcc = PipeCostConfig(fwd_time = 11,
                            bwd_time = [11]
                            )

    with pytest.raises(AssertionError):
        pcc = PipeCostConfig(fwd_time = [11,12],
                            bwd_time = [11],
                            wgt_time = [11])

    with pytest.raises(AssertionError):
        pcc = PipeCostConfig(fwd_time = [11],
                            bwd_time = 11,
                            wgt_time = 11)
                        
    # Valid

    pcc = PipeCostConfig(fwd_time = [2, 4, 6, 7, 8],
                        bwd_time = [4, 6, 7, 8, 10],
                        wgt_time = [3, 5, 7, 9, 11])
    
    assert pcc.fwd_time == [2, 4, 6, 7, 8]
    assert pcc.bwd_time == [4, 6, 7, 8, 10]
    assert pcc.wgt_time == [3, 5, 7, 9, 11]
    assert pcc.comm_time == 1
    
    pcc = PipeCostConfig(fwd_time = [[2, 4, 6, 7, 8], [2, 4, 6, 7, 8], [2, 4, 6, 7, 8]],
                        bwd_time = [[4, 6, 7, 8, 10], [4, 6, 7, 8, 10], [4, 6, 7, 8, 10]],
                        wgt_time = [[3, 5, 7, 9, 11], [3, 5, 7, 9, 11], [3, 5, 7, 9, 11]])
                        
    assert pcc.fwd_time == [[2, 4, 6, 7, 8], [2, 4, 6, 7, 8], [2, 4, 6, 7, 8]]
    assert pcc.bwd_time == [[4, 6, 7, 8, 10], [4, 6, 7, 8, 10], [4, 6, 7, 8, 10]]
    assert pcc.wgt_time == [[3, 5, 7, 9, 11], [3, 5, 7, 9, 11], [3, 5, 7, 9, 11]]
    assert pcc.comm_time == 1