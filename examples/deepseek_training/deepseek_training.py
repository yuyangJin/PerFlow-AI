'''
Example: DeepSeek Training Simulation with Configuration Search

This example demonstrates how to simulate DeepSeek training under various
pipeline parallel configurations and search for optimal settings.
'''

from perflowai.parallel.pipeline_parallel import (
    GPipeGraph,
    PipeDreamGraph,
    Interleaved1F1BGraph,
    ZeroBubbleGraph,
    ScheduleType,
    PipeCostConfig
)
from perflowai.simulator.pipeline import PPSimulator, PipeType
from perflowai.visualizer.trace_visualizer import TraceVisualizer


def simulate_deepseek_training(
    pipe_type,
    nstages=4,
    nmicrobatches=8,
    nchunks=1,
    cost_config=None,
    schedule_type=None
):
    """
    Simulate DeepSeek training with specified configuration.
    
    Args:
        pipe_type: Pipeline type (GPipe, PipeDream, Interleaved1F1B, ZeroBubble)
        nstages: Number of pipeline stages
        nmicrobatches: Number of microbatches
        nchunks: Number of chunks (for interleaved strategies)
        cost_config: Pipeline cost configuration
        schedule_type: Schedule type for ZeroBubble (ZB or ZBV)
    
    Returns:
        trace: Simulated trace
        perf: Performance metrics
    """
    # Default cost config based on DeepSeek-like workload characteristics
    if cost_config is None:
        # Simulated costs for a large language model like DeepSeek
        # Forward: 5000 time units, Backward: 10000 time units, Weight update: 3000 time units
        cost_config = PipeCostConfig(
            fwd_time=5000,
            bwd_time=10000,
            wgt_time=3000
        )
    
    # Build the appropriate graph based on pipeline type
    if pipe_type == PipeType.GPipe:
        graph = GPipeGraph(nstages, nmicrobatches, nchunks, cost_config=cost_config)
    elif pipe_type == PipeType.PipeDream:
        graph = PipeDreamGraph(nstages, nmicrobatches, nchunks, cost_config=cost_config)
    elif pipe_type == PipeType.Interleaved1F1B:
        graph = Interleaved1F1BGraph(nstages, nmicrobatches, nchunks, cost_config=cost_config)
    elif pipe_type == PipeType.ZeroBubble:
        if schedule_type is None:
            schedule_type = ScheduleType.ZB
        graph = ZeroBubbleGraph(
            nstages, 
            nmicrobatches, 
            nchunks, 
            cost_config=cost_config,
            schedule_type=schedule_type
        )
    else:
        raise ValueError(f"Unsupported pipeline type: {pipe_type}")
    
    # Build the computational graph
    graph.build_graph()
    
    # Run simulation
    simulator = PPSimulator(pipe_type, graph)
    trace = simulator.run()
    
    # Calculate performance metrics from trace
    makespan = 0
    for stage in range(trace.get_nstages()):
        stage_events = trace.get_events(stage)
        for event in stage_events:
            finish_time = event.get_timestamp() + event.get_duration()
            if finish_time > makespan:
                makespan = finish_time
    
    perf = {'makespan': makespan}
    
    return trace, perf


def search_optimal_configuration(
    nstages_options=[2, 4, 8],
    nmicrobatches_options=[4, 8, 16],
    nchunks_options=[1, 2, 4],
    visualize_best=True
):
    """
    Search for optimal configuration across different pipeline strategies.
    
    Args:
        nstages_options: List of stage counts to try
        nmicrobatches_options: List of microbatch counts to try
        nchunks_options: List of chunk counts to try
        visualize_best: Whether to visualize the best configuration
    
    Returns:
        best_config: Dictionary with best configuration details
        all_results: List of all tested configurations and their results
    """
    print("=" * 80)
    print("DeepSeek Training Configuration Search")
    print("=" * 80)
    
    # Cost configuration for DeepSeek-like workload
    cost_config = PipeCostConfig(
        fwd_time=5000,
        bwd_time=10000,
        wgt_time=3000
    )
    
    all_results = []
    best_config = {
        'makespan': float('inf'),
        'config': None,
        'trace': None
    }
    
    # Test different pipeline strategies
    strategies = [
        (PipeType.GPipe, "GPipe", False),
        (PipeType.PipeDream, "PipeDream", False),
        (PipeType.Interleaved1F1B, "Interleaved1F1B", True),
        (PipeType.ZeroBubble, "ZeroBubble", True),
    ]
    
    for pipe_type, strategy_name, supports_chunks in strategies:
        print(f"\n{'=' * 80}")
        print(f"Testing {strategy_name}")
        print(f"{'=' * 80}")
        
        for nstages in nstages_options:
            for nmicrobatches in nmicrobatches_options:
                # Determine valid chunk options
                if supports_chunks:
                    valid_chunks = nchunks_options
                else:
                    valid_chunks = [1]  # Only 1 chunk for GPipe and PipeDream
                
                for nchunks in valid_chunks:
                    # Skip invalid configurations
                    if supports_chunks and nchunks > 1:
                        # For interleaved strategies, need enough microbatches
                        if nmicrobatches < nstages * nchunks:
                            continue
                    
                    try:
                        # Run simulation
                        trace, perf = simulate_deepseek_training(
                            pipe_type=pipe_type,
                            nstages=nstages,
                            nmicrobatches=nmicrobatches,
                            nchunks=nchunks,
                            cost_config=cost_config,
                            schedule_type=ScheduleType.ZB if pipe_type == PipeType.ZeroBubble else None
                        )
                        
                        makespan = perf['makespan']
                        
                        # Store result
                        result = {
                            'strategy': strategy_name,
                            'pipe_type': pipe_type,
                            'nstages': nstages,
                            'nmicrobatches': nmicrobatches,
                            'nchunks': nchunks,
                            'makespan': makespan,
                            'trace': trace
                        }
                        all_results.append(result)
                        
                        # Print result
                        print(f"  Stages={nstages}, Microbatches={nmicrobatches}, Chunks={nchunks}: "
                              f"Makespan={makespan}")
                        
                        # Update best configuration
                        if makespan < best_config['makespan']:
                            best_config['makespan'] = makespan
                            best_config['config'] = result
                            best_config['trace'] = trace
                    
                    except Exception as e:
                        print(f"  Stages={nstages}, Microbatches={nmicrobatches}, Chunks={nchunks}: "
                              f"ERROR - {str(e)}")
    
    # Print best configuration
    print("\n" + "=" * 80)
    print("Best Configuration Found")
    print("=" * 80)
    best = best_config['config']
    print(f"Strategy: {best['strategy']}")
    print(f"Stages: {best['nstages']}")
    print(f"Microbatches: {best['nmicrobatches']}")
    print(f"Chunks: {best['nchunks']}")
    print(f"Makespan: {best['makespan']}")
    
    # Visualize best configuration
    if visualize_best and best_config['trace'] is not None:
        print("\nGenerating visualization for best configuration...")
        visualizer = TraceVisualizer(best_config['trace'])
        visualizer.visualize()
        print("Visualization saved as trace.svg")
    
    return best_config, all_results


def run_single_configuration_example():
    """
    Example: Run a single DeepSeek training simulation with specified configuration.
    """
    print("=" * 80)
    print("Example 1: Single Configuration Simulation")
    print("=" * 80)
    
    # Configuration for DeepSeek training
    nstages = 8
    nmicrobatches = 16
    nchunks = 2
    
    # Test different pipeline strategies
    strategies = [
        (PipeType.GPipe, "GPipe", 1),
        (PipeType.PipeDream, "PipeDream", 1),
        (PipeType.Interleaved1F1B, "Interleaved1F1B", nchunks),
        (PipeType.ZeroBubble, "ZeroBubble", nchunks),
    ]
    
    for pipe_type, strategy_name, chunks in strategies:
        print(f"\nTesting {strategy_name}...")
        trace, perf = simulate_deepseek_training(
            pipe_type=pipe_type,
            nstages=nstages,
            nmicrobatches=nmicrobatches,
            nchunks=chunks
        )
        print(f"  Makespan: {perf['makespan']}")


def run_configuration_search_example():
    """
    Example: Search for optimal configuration across different settings.
    """
    print("\n" + "=" * 80)
    print("Example 2: Configuration Search")
    print("=" * 80)
    
    # Search with smaller ranges for quick testing
    best_config, all_results = search_optimal_configuration(
        nstages_options=[4, 8],
        nmicrobatches_options=[8, 16],
        nchunks_options=[1, 2],
        visualize_best=True
    )
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    
    # Group by strategy
    from collections import defaultdict
    strategy_results = defaultdict(list)
    for result in all_results:
        strategy_results[result['strategy']].append(result['makespan'])
    
    for strategy, makespans in strategy_results.items():
        avg_makespan = sum(makespans) / len(makespans)
        min_makespan = min(makespans)
        max_makespan = max(makespans)
        print(f"\n{strategy}:")
        print(f"  Tested configurations: {len(makespans)}")
        print(f"  Average makespan: {avg_makespan:.2f}")
        print(f"  Best makespan: {min_makespan}")
        print(f"  Worst makespan: {max_makespan}")


if __name__ == "__main__":
    # Run single configuration example
    run_single_configuration_example()
    
    # Run configuration search example
    run_configuration_search_example()
