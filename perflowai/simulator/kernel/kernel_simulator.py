"""perflowai.simulator.kernel.kernel_simulator

Operator-level simulators.

Goal:
- Simulate a *single operator instance*'s compute throughput (FLOP/s) and peak memory (bytes).

Design:
- No dependency on events/tasks (the repo's Event.get_tasks() is not a stable interface).
- Use the FlowNode's inputs/outputs (Parameter) shapes/dtypes to infer workload.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Callable, Optional

from perflowai.core import DeviceConfig
from perflowai.util.checks import require
from perflowai.util.tensor import is_floating_dtype, numel, dtype_bytes
from perflowai.util.units import bandwidth_Bps
from perflowai.workflow.flow import FlowNode


class Parameter:
    def __init__(self, name: str, dtype: str, shape: tuple = None, value=None, trainable: bool = False):
        """
        :param name: The name or identifier of the parameter.
        :param dtype: The data type of the parameter (e.g., 'float32', 'int64').
        :param shape: The shape of the parameter (e.g., (3, 3) for a 3x3 matrix).
        :param value: The actual value of the parameter.
        :param trainable: Whether the parameter is trainable (default: False).
        """
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.value = value
        self.trainable = trainable

    def __repr__(self):
        return f"Parameter(name={self.name}, dtype={self.dtype}, shape={self.shape}, trainable={self.trainable})"

    def get_size(self) -> int:
        require(self.shape is not None, f"Parameter '{self.name}' must have shape")
        require(isinstance(self.shape, tuple), f"Parameter '{self.name}' shape must be tuple, got {type(self.shape)}")
        return numel(self.shape) * dtype_bytes(self.dtype)

    @property
    def id(self):
        return self.name

    @property
    def is_floating_point(self) -> bool:
        return is_floating_dtype(self.dtype)


@dataclass(frozen=True)
class KernelSimulationResult:
    flops: int
    bytes_accessed: int
    peak_memory_bytes: int
    compute_time_s: float
    memory_time_s: float
    time_s: float
    achieved_flops_per_s: float
    achieved_bytes_per_s: float
    bottleneck: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Workload:
    flops: int
    bytes_accessed: int
    peak_memory_bytes: int


class BaseKernelSimulator(FlowNode, ABC):
    """Base class for operator simulators driven by Parameters."""

    def __init__(
            self,
            name: str,
            id: int | str,
            inputs: list[Parameter],
            outputs: list[Parameter],
            device_config: DeviceConfig,
            compute_coeff: float = 0.6,
            memory_coeff: float = 0.8,
            sm_usage: float = 1.0,
            workload_aspect: Optional[Callable[["BaseKernelSimulator", Workload], Workload]] = None,
    ):
        """
        :param compute_coeff: device_config.compute_flops is scaled by this factor to model efficiency.
        :param memory_coeff: device_config.memory_bandwidth is scaled by this factor to model efficiency.
        :param sm_usage: Fraction of SM resources occupied (0.0=none, 1.0=full).
        :param workload_aspect: A function that takes (simulator, workload) and returns a modified Workload.

        final workload = workload_aspect(simulator, base workload) if workload_aspect is provided.
        """
        super().__init__(name=name, id=id, inputs=inputs, outputs=outputs)
        self.m_device_config = device_config
        self.compute_coeff = float(compute_coeff)
        self.memory_coeff = float(memory_coeff)
        self.sm_usage = float(sm_usage)
        self.workload_aspect = workload_aspect
        self._cached_workload: Optional[Workload] = None
        self._validate_common()
        self.validate_parameters()

    def _validate_common(self) -> None:
        require(self.compute_coeff > 0.0, "compute_coeff must be > 0")
        require(self.memory_coeff > 0.0, "memory_coeff must be > 0")
        require(self.m_device_config.compute_flops > 0, "device_config.compute_flops must be > 0")
        require(self.m_device_config.memory_bandwidth > 0, "device_config.memory_bandwidth must be > 0")

    @abstractmethod
    def validate_parameters(self) -> None:
        """Validate that inputs/outputs Parameters match this operator."""

    @property
    def is_tensor_op(self) -> bool:
        """Whether this kernel can utilize Tensor Cores."""
        return False

    def _get_peak_performance(self, dtype: str, bytes_accessed: int) -> tuple[float, float]:
        """
        Returns (peak_flops_s, bandwidth_Bps).
        Resolves MicroArchConfig if available, otherwise falls back to basic device_config.
        """
        microarch = self.m_device_config.microarch

        # Default Fallback (L0)
        peak_flops = float(self.m_device_config.compute_flops) * self.compute_coeff
        # Note: bandwidth_Bps uses memory_bandwidth (GB/s) -> B/s
        peak_bw = bandwidth_Bps(self.m_device_config) * self.memory_coeff

        if microarch is None:
            return peak_flops, peak_bw

        # 1. Refine Compute Peak
        # Try to find specific FLOPs/s from vector or tensor config
        flops_found = 0.0

        # Try tensor cores if eligible
        if self.is_tensor_op and microarch.compute.tensor.enabled:
            # keys are strings, e.g. "float16", "bf16"
            flops_dict = microarch.compute.tensor.peak_tflops_by_dtype
            if dtype in flops_dict:
                flops_found = flops_dict[dtype] * 1e12

        # Try vector cores (fallback or if not tensor op)
        if flops_found == 0.0:
            flops_dict = microarch.compute.vector.peak_tflops_by_dtype
            if dtype in flops_dict:
                flops_found = flops_dict[dtype] * 1e12

        if flops_found > 0.0:
            peak_flops = flops_found * self.compute_coeff

        # 2. Refine Memory Bandwidth (Hierarchical Cache Model)
        l1 = microarch.memory.l1
        l2 = microarch.memory.l2
        dram = microarch.memory.dram

        bw_found_gbps = 0.0

        # Check L1 (highest speed, smallest size)
        if l1.size_kb > 0 and bytes_accessed <= l1.size_kb * 1024:
            if l1.bandwidth_gbs > 0:
                bw_found_gbps = l1.bandwidth_gbs

        # Check L2
        if bw_found_gbps == 0.0 and l2.size_kb > 0 and bytes_accessed <= l2.size_kb * 1024:
            if l2.bandwidth_gbs > 0:
                bw_found_gbps = l2.bandwidth_gbs

        # DRAM fallthrough (or if L1/L2 bw not set / too small)
        if bw_found_gbps == 0.0:
            # If DRAM config has explicit bandwidth, use it. Otherwise keep using L0 fallback.
            if dram.bandwidth_gbs > 0:
                bw_found_gbps = dram.bandwidth_gbs

        if bw_found_gbps > 0.0:
            peak_bw = bw_found_gbps * 1e9 * self.memory_coeff

        return peak_flops, peak_bw

    @abstractmethod
    def _workload(self) -> tuple[int, int, int]:
        """Return (flops, bytes_accessed, peak_memory_bytes)."""

    def workload(self) -> Workload:
        if self._cached_workload is None:
            flops, bytes_accessed, peak_bytes = self._workload()
            if self.workload_aspect is not None:
                modified = self.workload_aspect(self, Workload(flops, bytes_accessed, peak_bytes))
                flops = modified.flops
                bytes_accessed = modified.bytes_accessed
                peak_bytes = modified.peak_memory_bytes
            self._cached_workload = Workload(flops, bytes_accessed, peak_bytes)
        return self._cached_workload

    def flops(self) -> int:
        return self.workload().flops

    def bytes_accessed(self) -> int:
        return self.workload().bytes_accessed

    def peak_memory_bytes(self) -> int:
        return self.workload().peak_memory_bytes

    # Backward/ergonomic aliases
    def get_memory_size_bytes(self) -> int:
        """Return this op's peak memory footprint in bytes."""

        return self.peak_memory_bytes()

    def get_peak_memory_bytes(self) -> int:
        """Alias of get_memory_size_bytes()."""

        return self.peak_memory_bytes()

    def simulate(self) -> KernelSimulationResult:
        """Simulate performance based on device peak compute/bandwidth.

        Returns an KernelSimulationResult with achieved FLOP/s and peak memory.
        """

        workload = self.workload()
        flops = workload.flops
        bytes_accessed = workload.bytes_accessed
        peak_bytes = workload.peak_memory_bytes

        # Infer dtype from first input
        dtype = self.m_inputs[0].dtype if self.m_inputs else "float32"

        peak_flops_s, peak_bw_bps = self._get_peak_performance(dtype, bytes_accessed)

        compute_time = float(flops) / peak_flops_s if peak_flops_s > 0 else 0.0
        memory_time = float(bytes_accessed) / peak_bw_bps if peak_bw_bps > 0 else 0.0
        time_s = max(compute_time, memory_time)

        bottleneck = "compute" if compute_time >= memory_time else "memory"
        achieved_flops = float(flops) / time_s if time_s > 0 else 0.0
        achieved_bw = float(bytes_accessed) / time_s if time_s > 0 else 0.0

        return KernelSimulationResult(
            flops=int(flops),
            bytes_accessed=int(bytes_accessed),
            peak_memory_bytes=int(peak_bytes),
            compute_time_s=compute_time,
            memory_time_s=memory_time,
            time_s=time_s,
            achieved_flops_per_s=achieved_flops,
            achieved_bytes_per_s=achieved_bw,
            bottleneck=bottleneck,
        )


class GEMMKernelSimulator(BaseKernelSimulator):
    """GEMM: C = A @ B

    Expected:
    - inputs:  A(m,k), B(k,n)
    - outputs: C(m,n) (optional; if absent, we infer its shape/dtype from A)
    """

    @property
    def is_tensor_op(self) -> bool:
        return True

    def __init__(
            self,
            device_config: DeviceConfig,
            a: Parameter,
            b: Parameter,
            c: Optional[Parameter] = None,
            *,
            name: str = "GEMM",
            id: int | str = 0,
            compute_coeff: float = 0.6,
            memory_coeff: float = 0.8,
            sm_usage: float = 1.0,
            workload_aspect: Optional[Callable[["BaseKernelSimulator", Workload], Workload]] = None,
    ):
        outputs = [c] if c is not None else []
        super().__init__(
            name=name,
            id=id,
            inputs=[a, b],
            outputs=outputs,
            device_config=device_config,
            compute_coeff=compute_coeff,
            memory_coeff=memory_coeff,
            sm_usage=sm_usage,
            workload_aspect=workload_aspect,
        )

    def validate_parameters(self) -> None:
        require(len(self.m_inputs) == 2, "GEMM requires exactly 2 inputs: A, B")
        a, b = self.m_inputs
        require(a.shape is not None and b.shape is not None, "GEMM inputs must have shapes")
        require(len(a.shape) == 2, f"GEMM A must be rank-2 (m,k), got {a.shape}")
        require(len(b.shape) == 2, f"GEMM B must be rank-2 (k,n), got {b.shape}")
        m, k_a = a.shape
        k_b, n = b.shape
        require(k_a == k_b, f"GEMM inner dim mismatch: A is (m={m},k={k_a}) but B is (k={k_b},n={n})")
        if len(self.m_outputs) == 1:
            c = self.m_outputs[0]
            require(c.shape is not None, "GEMM output C must have shape")
            require(c.shape == (m, n), f"GEMM output shape must be (m,n)=({m},{n}), got {c.shape}")

    def _workload(self) -> tuple[int, int, int]:
        a, b = self.m_inputs
        m, k = a.shape
        _, n = b.shape

        flops = 2 * m * n * k

        if len(self.m_outputs) == 1:
            c = self.m_outputs[0]
        else:
            c = Parameter(name="C", dtype=a.dtype, shape=(m, n))

        bytes_accessed = a.get_size() + b.get_size() + c.get_size()
        peak_bytes = a.get_size() + b.get_size() + c.get_size()
        return int(flops), int(bytes_accessed), int(peak_bytes)


class AttentionKernelSimulator(BaseKernelSimulator):
    """Scaled dot-product attention (simplified).

    Expected:
    - inputs:  Q(b,s,d), K(b,s,d), V(b,s,d)
    - outputs: O(b,s,d) (optional)

    FLOPs (approx, ignoring softmax exp/div):
    - QK^T: 2*b*h*s*s*head_dim
    - P*V:  2*b*h*s*s*head_dim
    Total: 4*b*h*s*s*head_dim
    """

    @property
    def is_tensor_op(self) -> bool:
        return True

    def __init__(
            self,
            device_config: DeviceConfig,
            q: Parameter,
            k: Parameter,
            v: Parameter,
            o: Optional[Parameter] = None,
            *,
            num_heads: int,
            name: str = "Attention",
            id: int | str = 0,
            compute_coeff: float = 0.6,
            memory_coeff: float = 0.8,
            sm_usage: float = 1.0,
            workload_aspect: Optional[Callable[["BaseKernelSimulator", Workload], Workload]] = None,
    ):
        self.num_heads = int(num_heads)
        outputs = [o] if o is not None else []
        super().__init__(
            name=name,
            id=id,
            inputs=[q, k, v],
            outputs=outputs,
            device_config=device_config,
            compute_coeff=compute_coeff,
            memory_coeff=memory_coeff,
            sm_usage=sm_usage,
            workload_aspect=workload_aspect,
        )

    def validate_parameters(self) -> None:
        require(self.num_heads > 0, "Attention num_heads must be > 0")
        require(len(self.m_inputs) == 3, "Attention requires exactly 3 inputs: Q, K, V")
        q, k, v = self.m_inputs
        require(q.shape is not None and k.shape is not None and v.shape is not None,
                "Attention inputs must have shapes")
        require(len(q.shape) == 3, f"Q must be rank-3 (b,s,d), got {q.shape}")
        require(len(k.shape) == 3, f"K must be rank-3 (b,s,d), got {k.shape}")
        require(len(v.shape) == 3, f"V must be rank-3 (b,s,d), got {v.shape}")
        require(q.shape == k.shape == v.shape, f"Q/K/V shapes must match, got Q={q.shape}, K={k.shape}, V={v.shape}")
        b, s, d = q.shape
        require(d % self.num_heads == 0, f"Hidden dim d={d} must be divisible by num_heads={self.num_heads}")
        if len(self.m_outputs) == 1:
            o = self.m_outputs[0]
            require(o.shape is not None, "Attention output O must have shape")
            require(o.shape == (b, s, d), f"Attention output must be (b,s,d)=({b},{s},{d}), got {o.shape}")

    def _workload(self) -> tuple[int, int, int]:
        q, k, v = self.m_inputs
        b, s, d = q.shape
        head_dim = d // self.num_heads
        flops = 4 * b * self.num_heads * s * s * head_dim

        if len(self.m_outputs) == 1:
            o = self.m_outputs[0]
        else:
            o = Parameter(name="O", dtype=q.dtype, shape=(b, s, d))

        bytes_accessed = q.get_size() + k.get_size() + v.get_size() + o.get_size()
        peak_bytes = q.get_size() + k.get_size() + v.get_size() + o.get_size()
        return int(flops), int(bytes_accessed), int(peak_bytes)


class Conv2dKernelSimulator(BaseKernelSimulator):
    """2D convolution (NCHW), simplified.

    Expected:
    - inputs:  X(n,c_in,h,w), W(c_out,c_in,k_h,k_w)
    - outputs: Y(n,c_out,h_out,w_out) (optional)
    """

    @property
    def is_tensor_op(self) -> bool:
        return True

    def __init__(
            self,
            device_config: DeviceConfig,
            x: Parameter,
            w: Parameter,
            y: Optional[Parameter] = None,
            *,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            name: str = "Conv2d",
            id: int | str = 0,
            compute_coeff: float = 0.6,
            memory_coeff: float = 0.8,
            sm_usage: float = 1.0,
            workload_aspect: Optional[Callable[["BaseKernelSimulator", Workload], Workload]] = None,
    ):
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        outputs = [y] if y is not None else []
        super().__init__(
            name=name,
            id=id,
            inputs=[x, w],
            outputs=outputs,
            device_config=device_config,
            compute_coeff=compute_coeff,
            memory_coeff=memory_coeff,
            sm_usage=sm_usage,
            workload_aspect=workload_aspect,
        )

    def _infer_out_hw(self, h: int, w: int, k_h: int, k_w: int) -> tuple[int, int]:
        require(self.stride > 0, "stride must be > 0")
        require(self.dilation > 0, "dilation must be > 0")
        require(self.padding >= 0, "padding must be >= 0")
        h_out = (h + 2 * self.padding - self.dilation * (k_h - 1) - 1) // self.stride + 1
        w_out = (w + 2 * self.padding - self.dilation * (k_w - 1) - 1) // self.stride + 1
        require(h_out > 0 and w_out > 0, f"Invalid output spatial size inferred: (h_out,w_out)=({h_out},{w_out})")
        return h_out, w_out

    def validate_parameters(self) -> None:
        require(len(self.m_inputs) == 2, "Conv2d requires exactly 2 inputs: X, W")
        x, w = self.m_inputs
        require(x.shape is not None and w.shape is not None, "Conv2d inputs must have shapes")
        require(len(x.shape) == 4, f"Conv2d X must be rank-4 NCHW, got {x.shape}")
        require(len(w.shape) == 4, f"Conv2d W must be rank-4 (c_out,c_in,k_h,k_w), got {w.shape}")
        n, c_in, h_in, w_in = x.shape
        c_out, c_in_w, k_h, k_w = w.shape
        require(c_in == c_in_w, f"Conv2d channel mismatch: X has c_in={c_in} but W expects c_in={c_in_w}")
        h_out, w_out = self._infer_out_hw(h_in, w_in, k_h, k_w)
        if len(self.m_outputs) == 1:
            y = self.m_outputs[0]
            require(y.shape is not None, "Conv2d output Y must have shape")
            require(
                y.shape == (n, c_out, h_out, w_out),
                f"Conv2d output must be (n,c_out,h_out,w_out)=({n},{c_out},{h_out},{w_out}), got {y.shape}",
            )

    def _workload(self) -> tuple[int, int, int]:
        x, w = self.m_inputs
        n, c_in, h_in, w_in = x.shape
        c_out, _, k_h, k_w = w.shape
        h_out, w_out = self._infer_out_hw(h_in, w_in, k_h, k_w)

        flops = 2 * n * c_out * h_out * w_out * c_in * k_h * k_w

        if len(self.m_outputs) == 1:
            y = self.m_outputs[0]
        else:
            y = Parameter(name="Y", dtype=x.dtype, shape=(n, c_out, h_out, w_out))

        bytes_accessed = x.get_size() + w.get_size() + y.get_size()
        peak_bytes = x.get_size() + w.get_size() + y.get_size()
        return int(flops), int(bytes_accessed), int(peak_bytes)


class SoftmaxKernelSimulator(BaseKernelSimulator):
    """Softmax over a single axis.

    Expected:
    - inputs:  X(...)
    - outputs: Y(same shape) (optional)

    Notes:
    - This is a *cost model*. exp/div are not true FLOPs on real hardware; we use a
    simple per-element approximation to get a stable relative estimate.
    """

    def __init__(
            self,
            device_config: DeviceConfig,
            x: Parameter,
            y: Optional[Parameter] = None,
            *,
            axis: int = -1,
            name: str = "Softmax",
            id: int | str = 0,
            compute_coeff: float = 0.6,
            memory_coeff: float = 0.8,
            workload_aspect: Optional[Callable[["BaseKernelSimulator", Workload], Workload]] = None,
    ):
        self.axis = int(axis)
        outputs = [y] if y is not None else []
        super().__init__(
            name=name,
            id=id,
            inputs=[x],
            outputs=outputs,
            device_config=device_config,
            compute_coeff=compute_coeff,
            memory_coeff=memory_coeff,
            workload_aspect=workload_aspect,
        )

    def validate_parameters(self) -> None:
        require(len(self.m_inputs) == 1, "Softmax requires exactly 1 input: X")
        x = self.m_inputs[0]
        require(x.shape is not None, "Softmax input X must have shape")
        require(isinstance(x.shape, tuple), f"Softmax input shape must be tuple, got {type(x.shape)}")
        require(len(x.shape) >= 1, f"Softmax input must have rank >= 1, got {x.shape}")

        rank = len(x.shape)
        axis = self.axis if self.axis >= 0 else self.axis + rank
        require(0 <= axis < rank, f"Softmax axis out of range: axis={self.axis}, rank={rank}")

        require(is_floating_dtype(x.dtype), f"Softmax expects floating dtype, got '{x.dtype}'")

        if len(self.m_outputs) == 1:
            y = self.m_outputs[0]
            require(y.shape is not None, "Softmax output Y must have shape")
            require(y.shape == x.shape, f"Softmax output shape must match input: X={x.shape}, Y={y.shape}")

    def _workload(self) -> tuple[int, int, int]:
        x = self.m_inputs[0]
        if len(self.m_outputs) == 1:
            y = self.m_outputs[0]
        else:
            y = Parameter(name="Y", dtype=x.dtype, shape=x.shape)

        rank = len(x.shape)
        axis = self.axis if self.axis >= 0 else self.axis + rank
        n = int(x.shape[axis])
        total = int(numel(x.shape))
        groups = total // n

        # Approx ops per group (stable heuristic):
        # - max reduce: (n-1)
        # - exp: n
        # - sum reduce: (n-1)
        # - div: n
        # => ~ (4n - 2)
        flops = groups * (4 * n - 2)

        x_bytes = x.get_size()
        y_bytes = y.get_size()
        require(x_bytes == y_bytes, "Softmax assumes output dtype/shape matches input")

        # Memory access model (heuristic):
        # read X, write temp(exp), read temp, write Y => ~4x tensor bytes
        bytes_accessed = 4 * x_bytes

        # Peak memory: X + temp + Y
        peak_bytes = x_bytes + x_bytes + y_bytes

        return int(flops), int(bytes_accessed), int(peak_bytes)
