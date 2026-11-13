"""Where the loader's time goes for one training sample of a pnpt arm: raw decode+resize
(LeRobotSource.__getitem__, no augmentation) vs the augmentation wrapper, per op.
Usage: .venv/bin/python prototypes/bench_data_path.py <run> [n_samples]
"""
import os, sys, time, statistics, random
import torch
from pathlib import Path
torch.set_num_threads(int(__import__("os").environ.get("BENCH_THREADS", "1")))
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from lewam.datasets import build_dataset
from lewam.utils.augment import Augmenter

run, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60
with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
    cfg = compose(config_name="base", overrides=[f"run={run}", f"dataset.root={os.environ.get('BENCH_ROOT', '<datasets_root>/b601_pnpt_gop1_256_200ep/')}"])
raw = build_dataset(cfg.dataset, augment=False, split="train", subsample=cfg.get("subsample_streams"))
aug_cfg = OmegaConf.to_container(cfg.dataset.augment, resolve=True)
full = Augmenter(aug_cfg["streams"], seed=0)
ops = {}
for pattern, spec in aug_cfg["streams"].items():
    for name, params in spec.items():
        ops[name] = Augmenter({pattern: {name: params}}, seed=0)
random.seed(0)
idxs = [random.randrange(len(raw)) for _ in range(n)]
def stats(xs): xs = sorted(xs); return f"mean {statistics.mean(xs)*1000:7.1f} ms  p50 {xs[len(xs)//2]*1000:7.1f}  p90 {xs[int(len(xs)*0.9)]*1000:7.1f}  max {xs[-1]*1000:7.1f}"
t_raw, t_full, t_ops = [], [], {k: [] for k in ops}
item = None
for i, idx in enumerate(idxs):
    t0 = time.perf_counter(); item = raw[idx]; t_raw.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); full(dict(item)); t_full.append(time.perf_counter() - t0)
    for k, a in ops.items():
        t0 = time.perf_counter(); a(dict(item)); t_ops[k].append(time.perf_counter() - t0)
    if i == 0:
        print("sample fields:", {k: tuple(v.shape) for k, v in item.items() if torch.is_tensor(v)})
print(f"samples: {n}  (raw index size {len(raw)})")
print("decode+resize (raw __getitem__):", stats(t_raw))
print("augment, all ops             :", stats(t_full))
for k, v in t_ops.items():
    print(f"  {k:16s}:", stats(v))
