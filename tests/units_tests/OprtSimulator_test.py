'''
test OprtSimulator class
'''


from perflowai.simulator import OprtSimulator
from perflowai.core import ModelConfig

def test_OprtSimulatorConfig():
    model_config = ModelConfig(
        name="test_model",
        url="https://example.com/model",
    )
    oprt_simulator = OprtSimulator(model_config)

    assert oprt_simulator.memory() == 0
    assert oprt_simulator.compute() == 0

    model_config_from_oprt_simulator = oprt_simulator.model_config()

    assert model_config_from_oprt_simulator.name == "test_model"
    assert model_config_from_oprt_simulator.url == "https://example.com/model"