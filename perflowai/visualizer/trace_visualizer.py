'''
@package perflowai.visualizer.trace_visualizer
'''


from perflowai.workflow.flow import FlowNode


'''
@class TraceVisiualizer
A trace visualizer.
'''
class TraceVisiualizer(FlowNode):
    def __init__(self, trace):
        self.trace = trace

    def visualize(self):
        # visualize the trace
        pass

    def run(self):
        self.visualize()
        pass