"""GEMM simulator example.

Sweep GEMM shapes by varying (m, k, n), record:
- achieved FLOP/s (from GEMMKernelSimulator)
- peak memory (bytes)

Then plot a dual-y-axis chart:
- x-axis: matrix size (we sweep square GEMMs: m=k=n=size)
- left y-axis: achieved throughput (TFLOP/s)
- right y-axis: peak memory (MiB)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from perflowai.core.device import DeviceConfig, DeviceType
from perflowai.simulator.kernel.kernel_simulator import GEMMKernelSimulator, Parameter


@dataclass(frozen=True)
class GemmPoint:
	m: int
	k: int
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


def sweep_square_gemm(
	sizes: Iterable[int],
	*,
	device: DeviceConfig,
	dtype: str = "float16",
	compute_coeff: float = 0.6,
	memory_coeff: float = 0.8,
) -> list[GemmPoint]:
	points: list[GemmPoint] = []
	for size in sizes:
		m = k = n = int(size)
		a = Parameter(name="A", dtype=dtype, shape=(m, k))
		b = Parameter(name="B", dtype=dtype, shape=(k, n))
		sim = GEMMKernelSimulator(
			device,
			a=a,
			b=b,
			compute_coeff=compute_coeff,
			memory_coeff=memory_coeff,
		)
		r = sim.simulate()
		points.append(
			GemmPoint(
				m=m,
				k=k,
				n=n,
				achieved_flops_per_s=float(r.achieved_flops_per_s),
				peak_memory_bytes=int(r.peak_memory_bytes),
			)
		)
	return points


def plot_dual_axis(points: list[GemmPoint]) -> None:
	import importlib

	try:
		plt = importlib.import_module("matplotlib.pyplot")
	except ModuleNotFoundError as e:
		raise ModuleNotFoundError(
			"matplotlib is required for plotting. Install with: pip install matplotlib"
		) from e

	x_sizes = [p.m for p in points]
	tflops = [p.achieved_flops_per_s / 1e12 for p in points]
	peak_mib = [p.peak_memory_bytes / (1024**2) for p in points]

	fig, ax1 = plt.subplots(figsize=(9, 5))
	ax2 = ax1.twinx()

	l1 = ax1.plot(x_sizes, tflops, marker="o", linewidth=1.5, label="Throughput")
	l2 = ax2.plot(x_sizes, peak_mib, marker="s", linewidth=1.5, label="Peak memory")

	ax1.set_xlabel("Matrix size (m=k=n)")
	ax1.set_ylabel("Throughput (TFLOP/s)")
	ax2.set_ylabel("Peak memory (MiB)")
	ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

	lines = l1 + l2
	labels = [ln.get_label() for ln in lines]
	ax1.legend(lines, labels, loc="best")
	ax1.set_title("GEMM simulated throughput and peak memory")

	fig.tight_layout()
	plt.show()


def main() -> None:
	device = default_device()

	# Square GEMM sweep; adjust this list as needed.
	sizes = [256, 512, 1024, 1536, 2048, 3072, 4096]

	points = sweep_square_gemm(
		sizes,
		device=device,
		dtype="float16",
		compute_coeff=0.6,
		memory_coeff=0.8,
	)

	print("m,k,n\tachieved_flops_per_s\tpeak_memory_bytes")
	for p in points:
		print(f"{p.m},{p.k},{p.n}\t{p.achieved_flops_per_s:.3e}\t{p.peak_memory_bytes}")

	plot_dual_axis(points)


if __name__ == "__main__":
	main()
