"""Softmax simulator example.

Sweep softmax length n (axis=-1) on a 2D tensor (b, n), record:
- achieved FLOP/s (from SoftmaxKernelSimulator)
- peak memory (bytes)

Then plot a dual-y-axis chart:
- x-axis: softmax length (n)
- left y-axis: achieved throughput (GFLOP/s)
- right y-axis: peak memory (MiB)

Notes:
- Softmax FLOPs here are a heuristic cost model; interpret throughput relatively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from perflowai.core.device import DeviceConfig, DeviceType
from perflowai.simulator.kernel.kernel_simulator import Parameter, SoftmaxKernelSimulator


@dataclass(frozen=True)
class SoftmaxPoint:
	b: int
	n: int
	achieved_flops_per_s: float
	peak_memory_bytes: int


def default_device() -> DeviceConfig:
	# Note: memory_bandwidth is GB/s in DeviceConfig; compute_flops is FLOP/s.
	return DeviceConfig(
		id=0,
		type=DeviceType.GPU,
		memory_capacity=80_000,
		memory_bandwidth=1000.0,
		compute_flops=100e12,
	)


def sweep_softmax_length(
	lengths: Iterable[int],
	*,
	device: DeviceConfig,
	batch: int = 4096,
	dtype: str = "float16",
	compute_coeff: float = 0.6,
	memory_coeff: float = 0.8,
) -> list[SoftmaxPoint]:
	points: list[SoftmaxPoint] = []
	b = int(batch)
	for n in lengths:
		n = int(n)
		x = Parameter(name="X", dtype=dtype, shape=(b, n))
		sim = SoftmaxKernelSimulator(
			device,
			x=x,
			axis=-1,
			compute_coeff=compute_coeff,
			memory_coeff=memory_coeff,
		)
		r = sim.simulate()
		points.append(
			SoftmaxPoint(
				b=b,
				n=n,
				achieved_flops_per_s=float(r.achieved_flops_per_s),
				peak_memory_bytes=int(r.peak_memory_bytes),
			)
		)
	return points


def plot_dual_axis(points: list[SoftmaxPoint]) -> None:
	import importlib

	try:
		plt = importlib.import_module("matplotlib.pyplot")
	except ModuleNotFoundError as e:
		raise ModuleNotFoundError(
			"matplotlib is required for plotting. Install with: pip install matplotlib"
		) from e

	x_n = [p.n for p in points]
	gflops = [p.achieved_flops_per_s / 1e9 for p in points]
	peak_mib = [p.peak_memory_bytes / (1024**2) for p in points]

	fig, ax1 = plt.subplots(figsize=(9, 5))
	ax2 = ax1.twinx()

	l1 = ax1.plot(x_n, gflops, marker="o", linewidth=1.5, label="Throughput")
	l2 = ax2.plot(x_n, peak_mib, marker="s", linewidth=1.5, label="Peak memory")

	ax1.set_xlabel("Softmax length (n)")
	ax1.set_ylabel("Throughput (GFLOP/s)")
	ax2.set_ylabel("Peak memory (MiB)")
	ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

	lines = l1 + l2
	labels = [ln.get_label() for ln in lines]
	ax1.legend(lines, labels, loc="best")
	ax1.set_title("Softmax simulated throughput and peak memory")

	fig.tight_layout()
	plt.show()


def main() -> None:
	device = default_device()

	# Softmax length sweep; adjust this list as needed.
	lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

	points = sweep_softmax_length(
		lengths,
		device=device,
		batch=4096,
		dtype="float16",
		compute_coeff=0.6,
		memory_coeff=0.8,
	)

	print("b,n\tachieved_flops_per_s\tpeak_memory_bytes")
	for p in points:
		print(f"{p.b},{p.n}\t{p.achieved_flops_per_s:.3e}\t{p.peak_memory_bytes}")

	plot_dual_axis(points)


if __name__ == "__main__":
	main()
