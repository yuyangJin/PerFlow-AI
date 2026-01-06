"""Attention simulator example.

Sweep attention sequence length s, record:
- achieved FLOP/s (from AttentionKernelSimulator)
- peak memory (bytes)

Then plot a dual-y-axis chart:
- x-axis: sequence length (s)
- left y-axis: achieved throughput (TFLOP/s)
- right y-axis: peak memory (MiB)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from perflowai.core.device import DeviceConfig, DeviceType
from perflowai.simulator.kernel.kernel_simulator import AttentionKernelSimulator, Parameter


@dataclass(frozen=True)
class AttentionPoint:
	b: int
	s: int
	d: int
	num_heads: int
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


def sweep_sequence_length(
	seq_lens: Iterable[int],
	*,
	device: DeviceConfig,
	batch: int = 1,
	hidden_dim: int = 4096,
	num_heads: int = 32,
	dtype: str = "float16",
	compute_coeff: float = 0.6,
	memory_coeff: float = 0.8,
) -> list[AttentionPoint]:
	points: list[AttentionPoint] = []
	b = int(batch)
	d = int(hidden_dim)
	for s in seq_lens:
		s = int(s)
		q = Parameter(name="Q", dtype=dtype, shape=(b, s, d))
		k = Parameter(name="K", dtype=dtype, shape=(b, s, d))
		v = Parameter(name="V", dtype=dtype, shape=(b, s, d))
		sim = AttentionKernelSimulator(
			device,
			q=q,
			k=k,
			v=v,
			num_heads=num_heads,
			compute_coeff=compute_coeff,
			memory_coeff=memory_coeff,
		)
		r = sim.simulate()
		points.append(
			AttentionPoint(
				b=b,
				s=s,
				d=d,
				num_heads=int(num_heads),
				achieved_flops_per_s=float(r.achieved_flops_per_s),
				peak_memory_bytes=int(r.peak_memory_bytes),
			)
		)
	return points


def plot_dual_axis(points: list[AttentionPoint]) -> None:
	import importlib

	try:
		plt = importlib.import_module("matplotlib.pyplot")
	except ModuleNotFoundError as e:
		raise ModuleNotFoundError(
			"matplotlib is required for plotting. Install with: pip install matplotlib"
		) from e

	x_seq = [p.s for p in points]
	tflops = [p.achieved_flops_per_s / 1e12 for p in points]
	peak_mib = [p.peak_memory_bytes / (1024**2) for p in points]

	fig, ax1 = plt.subplots(figsize=(9, 5))
	ax2 = ax1.twinx()

	l1 = ax1.plot(x_seq, tflops, marker="o", linewidth=1.5, label="Throughput")
	l2 = ax2.plot(x_seq, peak_mib, marker="s", linewidth=1.5, label="Peak memory")

	ax1.set_xlabel("Sequence length (s)")
	ax1.set_ylabel("Throughput (TFLOP/s)")
	ax2.set_ylabel("Peak memory (MiB)")
	ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

	lines = l1 + l2
	labels = [ln.get_label() for ln in lines]
	ax1.legend(lines, labels, loc="best")
	ax1.set_title("Attention simulated throughput and peak memory")

	fig.tight_layout()
	plt.show()


def main() -> None:
	device = default_device()

	# Sequence length sweep; adjust this list as needed.
	seq_lens = [128, 256, 512, 1024, 1536, 2048, 3072, 4096]

	points = sweep_sequence_length(
		seq_lens,
		device=device,
		batch=1,
		hidden_dim=4096,
		num_heads=32,
		dtype="float16",
		compute_coeff=0.6,
		memory_coeff=0.8,
	)

	print("b,s,d,heads\tachieved_flops_per_s\tpeak_memory_bytes")
	for p in points:
		print(f"{p.b},{p.s},{p.d},{p.num_heads}\t{p.achieved_flops_per_s:.3e}\t{p.peak_memory_bytes}")

	plot_dual_axis(points)


if __name__ == "__main__":
	main()
