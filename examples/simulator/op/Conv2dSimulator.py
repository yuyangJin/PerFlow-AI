"""Conv2d simulator example.

Sweep input spatial size (H=W), record:
- achieved FLOP/s (from Conv2dKernelSimulator)
- peak memory (bytes)

Then plot a dual-y-axis chart:
- x-axis: spatial size (H=W)
- left y-axis: achieved throughput (TFLOP/s)
- right y-axis: peak memory (MiB)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from perflowai.core.device import DeviceConfig, DeviceType
from perflowai.simulator.kernel.kernel_simulator import Conv2dKernelSimulator, Parameter


@dataclass(frozen=True)
class Conv2dPoint:
	n: int
	c_in: int
	c_out: int
	h: int
	w: int
	k_h: int
	k_w: int
	stride: int
	padding: int
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


def sweep_spatial_size(
	sizes: Iterable[int],
	*,
	device: DeviceConfig,
	n: int = 1,
	c_in: int = 64,
	c_out: int = 128,
	k_h: int = 3,
	k_w: int = 3,
	stride: int = 1,
	padding: int = 1,
	dilation: int = 1,
	dtype: str = "float16",
	compute_coeff: float = 0.6,
	memory_coeff: float = 0.8,
) -> list[Conv2dPoint]:
	points: list[Conv2dPoint] = []
	for size in sizes:
		h = w = int(size)
		x = Parameter(name="X", dtype=dtype, shape=(int(n), int(c_in), h, w))
		wgt = Parameter(name="W", dtype=dtype, shape=(int(c_out), int(c_in), int(k_h), int(k_w)))
		sim = Conv2dKernelSimulator(
			device,
			x=x,
			w=wgt,
			stride=int(stride),
			padding=int(padding),
			dilation=int(dilation),
			compute_coeff=compute_coeff,
			memory_coeff=memory_coeff,
		)
		r = sim.simulate()
		points.append(
			Conv2dPoint(
				n=int(n),
				c_in=int(c_in),
				c_out=int(c_out),
				h=h,
				w=w,
				k_h=int(k_h),
				k_w=int(k_w),
				stride=int(stride),
				padding=int(padding),
				achieved_flops_per_s=float(r.achieved_flops_per_s),
				peak_memory_bytes=int(r.peak_memory_bytes),
			)
		)
	return points


def plot_dual_axis(points: list[Conv2dPoint]) -> None:
	import importlib

	try:
		plt = importlib.import_module("matplotlib.pyplot")
	except ModuleNotFoundError as e:
		raise ModuleNotFoundError(
			"matplotlib is required for plotting. Install with: pip install matplotlib"
		) from e

	x_sizes = [p.h for p in points]
	tflops = [p.achieved_flops_per_s / 1e12 for p in points]
	peak_mib = [p.peak_memory_bytes / (1024**2) for p in points]

	fig, ax1 = plt.subplots(figsize=(9, 5))
	ax2 = ax1.twinx()

	l1 = ax1.plot(x_sizes, tflops, marker="o", linewidth=1.5, label="Throughput")
	l2 = ax2.plot(x_sizes, peak_mib, marker="s", linewidth=1.5, label="Peak memory")

	ax1.set_xlabel("Spatial size (H=W)")
	ax1.set_ylabel("Throughput (TFLOP/s)")
	ax2.set_ylabel("Peak memory (MiB)")
	ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

	lines = l1 + l2
	labels = [ln.get_label() for ln in lines]
	ax1.legend(lines, labels, loc="best")
	ax1.set_title("Conv2d simulated throughput and peak memory")

	fig.tight_layout()
	plt.show()


def main() -> None:
	device = default_device()

	# Spatial sweep; adjust this list as needed.
	sizes = [32, 48, 64, 96, 128, 160, 192, 224]

	points = sweep_spatial_size(
		sizes,
		device=device,
		n=1,
		c_in=64,
		c_out=128,
		k_h=3,
		k_w=3,
		stride=1,
		padding=1,
		dilation=1,
		dtype="float16",
		compute_coeff=0.6,
		memory_coeff=0.8,
	)

	print("n,c_in,c_out,h,w,k_h,k_w,stride,pad\tachieved_flops_per_s\tpeak_memory_bytes")
	for p in points:
		print(
			f"{p.n},{p.c_in},{p.c_out},{p.h},{p.w},{p.k_h},{p.k_w},{p.stride},{p.padding}"
			f"\t{p.achieved_flops_per_s:.3e}\t{p.peak_memory_bytes}"
		)

	plot_dual_axis(points)


if __name__ == "__main__":
	main()
