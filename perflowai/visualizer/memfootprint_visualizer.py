'''
@module memory footprint visualizer
'''

import matplotlib.pyplot as plt 

from ..workflow import FlowNode

'''
@class MemoryFootprintVisualizer
'''

class MemoryFootprintVisualizer(FlowNode):
    def __init__(self, mem_fp):
        self.mem_fp = mem_fp

    def visualize(self):
        # Plotting the memory footprint for the current stage  
        plt.figure(figsize=(10, 6))  
        
        for stage_id, memory_usage in self.mem_fp.items():

            # Unzip the time and memory usage for plotting  
            times, memory_levels = zip(*memory_usage)

            plt.plot(times, memory_levels, marker='o', label = f'Stage {stage_id}')  

        plt.title(f'Memory Usage Over Timeline')  
        plt.xlabel('Time')  
        plt.ylabel('Memory Usage')  
        plt.legend()
        plt.grid()  
        plt.savefig('memory_footprint.pdf', bbox_inches='tight')  