'''
test PipePartitionConfig
'''

import pytest

from perflowai.simulator.pp_simulator import PipePartitionConfig

def test_PipePartitionConfig():
    with pytest.raises(AssertionError):
        ppc = PipePartitionConfig(4, [0.1, 0.2, 0.3])

    with pytest.raises(AssertionError):
        ppc = PipePartitionConfig(4, [0.1, 0.2, 0.3, 0.3])


    ppc = PipePartitionConfig(4, [0.1, 0.2, 0.3, 0.4])

    assert ppc.get_nstages() == 4
    assert ppc.get_partition_plan() == [0.1, 0.2, 0.3, 0.4]
