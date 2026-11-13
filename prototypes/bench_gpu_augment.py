"""Does GPU augmentation help? Self-contained: synthetic uint8 windows shaped like a
pnpt training sample (two cameras x 5 frames x 3 x 256 x 256) run through the b601
augmentation recipe on one CPU thread (a DataLoader worker's budget) and on the GPU,
per sample and per micro-batch. No dataset, no hydra -- only lewam.utils.augment.

    .venv/bin/python prototypes/bench_gpu_augment.py [n_samples] [batch] [--frames 5] [--size 256]

The ops and parameters are configs/dataset/b601_pnpt.yaml's `augment.streams`; pass
--run <arm> to read them from a composed config instead.
"""
import argparse, statistics, sys, time
from pathlib import Path

import torch

from lewam.utils.augment import Augmenter

B601_AUGMENT = {
    "observation.images.*": {
        "random_crop": {"scale": [0.875, 1.0]},
        "rotation": {"degrees": 7.5, "p": 0.5},
        "color_jitter": {"brightness": 0.2, "contrast": 0.2, "gamma": 0.2, "hue": 0.02},
        "gaussian_noise": {"std": 0.005},
    }
}
FIELDS = ["observation.images.zed_left", "observation.images.wrist_cam"]

ap = argparse.ArgumentParser()
ap.add_argument("n", nargs="?", type=int, default=64)
ap.add_argument("batch", nargs="?", type=int, default=32)
ap.add_argument("--frames", type=int, default=5)
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--run", default=None, help="read augment.streams from this run config")
args = ap.parse_args()
torch.set_num_threads(1)

streams = B601_AUGMENT
if args.run:
    import fnmatch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
                               version_base=None):
        cfg = compose(config_name="base", overrides=[f"run={args.run}"])
    streams = OmegaConf.to_container(cfg.dataset.augment, resolve=True)["streams"]
    streams = {p: o for p, o in streams.items() if any(fnmatch.fnmatchcase(f, p) for f in FIELDS)}
print(f"ops: { {p: list(o) for p, o in streams.items()} }")
print(f"sample: {len(FIELDS)} cameras x ({args.frames}, 3, {args.size}, {args.size}) uint8")

g = torch.Generator().manual_seed(0)
items = [{f: torch.randint(0, 256, (args.frames, 3, args.size, args.size), generator=g, dtype=torch.uint8)
          for f in FIELDS} for _ in range(args.n)]

def stat(xs):
    xs = sorted(xs)
    return (f"mean {statistics.mean(xs)*1e3:7.2f} ms  p50 {xs[len(xs)//2]*1e3:7.2f}  "
            f"p90 {xs[int(len(xs)*.9)]*1e3:7.2f}  max {xs[-1]*1e3:7.2f}")

def per_op(streams):
    return {name: Augmenter({p: {name: params}}, seed=0)
            for p, ops in streams.items() for name, params in ops.items()}

cpu_aug, cpu_ops = Augmenter(streams, seed=0), per_op(streams)
t_cpu, t_cpu_ops = [], {k: [] for k in cpu_ops}
for it in items:
    t0 = time.perf_counter(); cpu_aug(dict(it)); t_cpu.append(time.perf_counter() - t0)
    for k, a in cpu_ops.items():
        t0 = time.perf_counter(); a(dict(it)); t_cpu_ops[k].append(time.perf_counter() - t0)
print("\nCPU, 1 thread, per sample (all ops)     :", stat(t_cpu))
for k, v in t_cpu_ops.items():
    print(f"  {k:16s}                      :", stat(v))

if not torch.cuda.is_available():
    print("\nno CUDA device here: run on a GPU box for the GPU rows")
    sys.exit(0)
dev = torch.device("cuda")
gpu_aug, gpu_ops = Augmenter(streams, seed=0), per_op(streams)
for it in items[:3]:                                   # warm-up: kernels, allocator
    gpu_aug({k: v.to(dev) for k, v in it.items()})
torch.cuda.synchronize()

t_h2d, t_gpu, t_gpu_ops = [], [], {k: [] for k in gpu_ops}
for it in items:
    torch.cuda.synchronize(); t0 = time.perf_counter()
    on_dev = {k: v.to(dev) for k, v in it.items()}
    torch.cuda.synchronize(); t1 = time.perf_counter()
    gpu_aug(dict(on_dev))
    torch.cuda.synchronize(); t2 = time.perf_counter()
    t_h2d.append(t1 - t0); t_gpu.append(t2 - t1)
    for k, a in gpu_ops.items():
        torch.cuda.synchronize(); t0 = time.perf_counter()
        a(dict(on_dev))
        torch.cuda.synchronize(); t_gpu_ops[k].append(time.perf_counter() - t0)
print("\nhost->device copy per sample            :", stat(t_h2d))
print("GPU, per sample (all ops)               :", stat(t_gpu))
for k, v in t_gpu_ops.items():
    print(f"  {k:16s}                      :", stat(v))

bs = args.batch
batches = [items[i:i + bs] for i in range(0, len(items), bs) if len(items[i:i + bs]) == bs]
t_batch = []
for b in batches:
    stacked = {f: torch.stack([it[f] for it in b]).to(dev) for f in FIELDS}
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for j in range(bs):                                # one draw per sample, as the workers do
        gpu_aug({f: stacked[f][j] for f in FIELDS})
    torch.cuda.synchronize(); t_batch.append(time.perf_counter() - t0)
if t_batch:
    print(f"GPU, micro-batch of {bs} with per-sample draws:", stat(t_batch),
          f"-> {statistics.mean(t_batch)/bs*1e3:.2f} ms per sample")
print(f"\nGPU is {statistics.mean(t_cpu)/statistics.mean(t_gpu):.1f}x faster per sample than one CPU worker "
      f"thread; a micro-batch of {bs} costs "
      f"{statistics.mean(t_batch)*1e3 if t_batch else float('nan'):.0f} ms of GPU time versus "
      f"{statistics.mean(t_cpu)*bs*1e3/8:.0f} ms of wall time spread over 8 CPU workers.")
