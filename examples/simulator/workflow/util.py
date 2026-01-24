from perflowai.simulator.kernel.kernel_simulator import Parameter


def flatten_param(p: Parameter) -> Parameter:
    """Returns a view of the parameter as rank-2 (flattening leading dims), sharing the same name/id."""
    if p.shape and len(p.shape) == 3:
        b, s, d = p.shape
        # Intentionally keep the same name so the memory manager treats it as the same buffer.
        return Parameter(name=p.name, dtype=p.dtype, shape=(b * s, d))
    return p
