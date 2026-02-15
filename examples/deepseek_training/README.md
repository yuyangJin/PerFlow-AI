# DeepSeek Training Simulation Example

This example demonstrates how to simulate DeepSeek training under various pipeline parallel configurations and search for optimal settings using PerFlow-AI.

## Features

- **Multiple Pipeline Strategies**: Support for GPipe, PipeDream, Interleaved1F1B, and ZeroBubble strategies
- **Configuration Search**: Automatically search for optimal configurations across different parameters
- **Performance Comparison**: Compare performance metrics across different strategies
- **Visualization**: Generate visual traces of the best performing configurations

## Usage

### Run the Complete Example

Run all examples including single configuration and configuration search:

```bash
cd examples/deepseek_training
python deepseek_training.py
```

### Run Single Configuration Example

Test multiple pipeline strategies with a fixed configuration:

```python
from deepseek_training import run_single_configuration_example

run_single_configuration_example()
```

### Run Configuration Search

Search for the optimal configuration:

```python
from deepseek_training import run_configuration_search_example

run_configuration_search_example()
```

### Custom Simulation

Simulate DeepSeek training with custom parameters:

```python
from deepseek_training import simulate_deepseek_training
from perflowai.simulator.pipeline import PipeType
from perflowai.parallel.pipeline_parallel import PipeCostConfig

# Custom cost configuration
cost_config = PipeCostConfig(
    fwd_time=5000,   # Forward pass time
    bwd_time=10000,  # Backward pass time
    wgt_time=3000    # Weight update time
)

# Run simulation
trace, perf = simulate_deepseek_training(
    pipe_type=PipeType.ZeroBubble,
    nstages=8,
    nmicrobatches=16,
    nchunks=2,
    cost_config=cost_config
)

print(f"Makespan: {perf['makespan']}")
```

### Custom Configuration Search

Search with custom parameter ranges:

```python
from deepseek_training import search_optimal_configuration

best_config, all_results = search_optimal_configuration(
    nstages_options=[2, 4, 8, 16],
    nmicrobatches_options=[4, 8, 16, 32],
    nchunks_options=[1, 2, 4],
    visualize_best=True
)
```

## Configuration Parameters

### Pipeline Strategies

- **GPipe**: Synchronous pipeline with periodic flushes
- **PipeDream**: Asynchronous pipeline with 1F1B (1 Forward, 1 Backward) schedule
- **Interleaved1F1B**: Interleaved 1F1B with multiple model chunks per device
- **ZeroBubble**: Advanced schedule that minimizes pipeline bubbles

### Key Parameters

- `nstages`: Number of pipeline stages (typically 2-16)
- `nmicrobatches`: Number of microbatches (typically 4-32)
- `nchunks`: Number of model chunks per device (1, 2, or 4 for interleaved strategies)
- `cost_config`: Performance characteristics (forward, backward, weight update times)

## Output

The script outputs:

1. **Performance Metrics**: Makespan for each configuration
2. **Best Configuration**: Optimal strategy and parameters found
3. **Summary Statistics**: Average, best, and worst performance per strategy
4. **Visualization**: `trace.svg` file showing the execution trace of the best configuration

## Example Output

```
================================================================================
Example 1: Single Configuration Simulation
================================================================================

Testing GPipe...
  Makespan: 1040000

Testing PipeDream...
  Makespan: 360000

Testing Interleaved1F1B...
  Makespan: 296000

Testing ZeroBubble...
  Makespan: 276000

================================================================================
Example 2: Configuration Search
================================================================================
DeepSeek Training Configuration Search
================================================================================

================================================================================
Testing GPipe
================================================================================
  Stages=4, Microbatches=8, Chunks=1: Makespan=520000
  Stages=4, Microbatches=16, Chunks=1: Makespan=1040000
  ...

================================================================================
Best Configuration Found
================================================================================
Strategy: ZeroBubble
Stages: 8
Microbatches: 16
Chunks: 2
Makespan: 276000
```

## Understanding Results

- **Lower makespan is better**: Indicates faster training completion
- **Strategy comparison**: Different strategies have different bubble ratios and efficiency
- **Configuration impact**: More stages or chunks can reduce bubbles but add overhead
- **Optimal configuration**: Depends on model size, hardware, and workload characteristics

## Notes

- The default cost configuration uses simulated values for DeepSeek-like workloads
- Actual training times depend on model architecture, hardware, and batch size
- The simulation assumes ideal network communication and no other overhead
- Use this tool to explore trade-offs before running actual distributed training
