"""How long one `model.predict()` takes on this machine, for any run arm.

Answers "can this box close the loop?": a replan must finish inside the wall-clock span
of the chunk it produces, or the arm runs out of actions before the next one arrives.
Prints that comparison plus a breakdown of where the time goes, since for the flow arms
the frozen V-JEPA encoder is most of it and the flow steps are a rounding error.

Runs on random weights and random inputs -- shapes and FLOPs are what set the time, so
the numbers match a trained checkpoint. `--pretrained` loads the real encoder weights
instead (a 1.6GB download for vit_base on first use), which changes nothing but the wait.

    python src/lewam/debug/bench_inference.py
    python src/lewam/debug/bench_inference.py flow_wam_b601_pnpt --device mps
    python src/lewam/debug/bench_inference.py --device mps --dtype fp16 --compile

Device is named, not detected: `cpu` is the default because it is the number comparable
across boxes, and it is what a robot box without a GPU will actually pay.

`--dtype fp16` is the win on MPS and CUDA. On CPU it is usually a LOSS -- torch has no
half-precision CPU kernels for much of this and falls back through float32 conversions --
so `--dtype bf16` is the one to try there.

Reduced precision runs through autocast, not by casting the weights: V-JEPA's rotary
embedding leaves q and k in float32 while v follows the weights, and half-cast weights
make attention raise on the mismatch. Autocast casts per op, so it matches what a
deployment would do anyway.

`--compile` compiles the two modules the time actually goes to (the V-JEPA backbone and
the trunk), not the whole model: `predict()` is a Python-level integration loop that
Dynamo would trace through rather than speed up. Compilation happens on the first call
and is reported separately from the timed median.
"""

import argparse
import contextlib
import os
import statistics
import time
from pathlib import Path

# unimplemented MPS ops fall back to CPU rather than raising. Must be set before
# torch is imported; an explicit env var wins.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import hydra  # noqa: E402
import torch  # noqa: E402

from lewam.algorithms import build_algorithm  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

# native B601 dataset resolutions, so the measurement includes the resize to the
# encoder's crop_size that the real pipeline pays
IMAGE_SHAPES = {
    "observation.images.zed_left": (376, 672),
    "observation.images.zed_right": (376, 672),
    "observation.images.wrist_cam": (360, 640),
}
DEFAULT_IMAGE_SHAPE = (376, 672)

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", nargs="?", default="flow_policy_a2a_b601_towel_fold",
                   help="a config in configs/run/ (default: %(default)s)")
    p.add_argument("--iters", type=int, default=10, help="timed calls after a warmup (default: %(default)s)")
    p.add_argument("--device", default="cpu", help="cpu | mps | cuda (default: %(default)s)")
    p.add_argument("--dtype", default="fp32", choices=sorted(DTYPES),
                   help="model and input precision (default: %(default)s)")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the V-JEPA backbone and the trunk")
    p.add_argument("--pretrained", action="store_true",
                   help="load the real V-JEPA weights instead of random ones (slow first run, same timings)")
    p.add_argument("--image-size", metavar="H,W",
                   help="override every camera's input resolution, e.g. 256,256")
    return p.parse_args()


def compile_hot_modules(model, amp):
    """Compile the forward of the V-JEPA backbone and the trunk in place.

    Bound-method compilation, not module wrapping: both are reached from inside their
    caller (`self.backbone(x)`, `self(...)` in the Euler loop), and an OptimizedModule
    wrapper forwards attribute access back to the original, so a compiled wrapper would
    be bypassed on exactly the calls that matter.

    `amp` is entered INSIDE the compiled function rather than left ambient around it.
    Traced from the outside, the graph can end up with one addmm operand cast and the
    other not ("mat1 and mat2 must have the same dtype, but got Float and Half"); entering
    it within the traced region makes the casts part of the graph. Nesting is harmless
    when the caller holds an autocast too.
    """
    targets = []
    if "vjepa" in model.encoders:
        targets.append(model.encoders["vjepa"].backbone)
    targets.append(model.predictor)

    for module in targets:
        def under_amp(*args, _forward=module.forward, **kwargs):
            with amp():
                return _forward(*args, **kwargs)

        module.forward = torch.compile(under_amp)
    return [type(m).__name__ for m in targets]


def build_obs(model, cfg, device, image_size):
    """One random observation window per field the model conditions on, in the shapes
    the eval driver hands it: (1, T, C, H, W) uint8 for cameras, (1, T, dim) float32
    otherwise -- what the driver produces at every --dtype, since autocast casts per op
    rather than expecting pre-cast inputs."""
    visual = {cfg.algorithm.scene_field, cfg.algorithm.wrist_field}
    state_fields = list(cfg.algorithm.state_fields)
    state_width = int(cfg.algorithm.state_dim) // len(state_fields)

    obs = {}
    for field, offsets in model.field_offsets.items():
        steps = len(offsets)
        if field in visual:
            h, w = image_size or IMAGE_SHAPES.get(field, DEFAULT_IMAGE_SHAPE)
            obs[field] = torch.randint(0, 256, (1, steps, 3, h, w), dtype=torch.uint8, device=device)
        elif field in state_fields:
            obs[field] = torch.randn(1, steps, state_width, device=device)
        elif field == model.action_field or field == "action":
            obs[field] = torch.randn(1, steps, int(cfg.algorithm.action_dim), device=device)
        else:
            raise ValueError(
                f"no width known for conditioning field '{field}'; add it to build_obs()"
            )
    return obs


def autocaster(device, dtype):
    """A fresh autocast context per call, or a no-op at fp32. Fresh rather than one
    shared instance: an autocast context carries per-entry state."""
    if dtype is torch.float32:
        return contextlib.nullcontext
    return lambda: torch.autocast(device_type=device, dtype=dtype)


def timer(device):
    """Median seconds per call, with the device drained first so async work lands
    inside the measurement rather than the next one."""
    sync = (torch.mps.synchronize if device == "mps"
            else torch.cuda.synchronize if device == "cuda"
            else lambda: None)

    def run(fn, iters):
        fn()
        sync()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            sync()
            times.append(time.perf_counter() - t0)
        return sorted(times)

    return run


def main():
    args = parse_args()
    image_size = tuple(int(v) for v in args.image_size.split(",")) if args.image_size else None

    overrides = [f"run={args.run}"]
    if not args.pretrained:
        overrides.append("++algorithm.encoders.vjepa.checkpoint=null")
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        # only cfg.algorithm is read; the rest is left unresolved, since other sections
        # carry ${hydra:...} interpolations that need a live HydraConfig
        cfg = hydra.compose(config_name="base", overrides=overrides)

    dtype = DTYPES[args.dtype]
    built = time.perf_counter()
    model = build_algorithm(cfg.algorithm).to(args.device).eval()
    built = time.perf_counter() - built
    amp = autocaster(args.device, dtype)
    compiled = compile_hot_modules(model, amp) if args.compile else []

    params = sum(p.numel() for p in model.parameters())
    obs = build_obs(model, cfg, args.device, image_size)
    time_it = timer(args.device)

    print(f"run       {args.run}")
    print(f"device    {args.device}   dtype={args.dtype}   threads={torch.get_num_threads()}"
          f"   torch={torch.__version__}")
    print(f"compile   {', '.join(compiled) if compiled else 'off'}")
    print(f"model     {params / 1e6:.1f}M params, {model.num_flow_steps} flow steps, built in {built:.1f}s")
    print(f"windows   {({f: len(o) for f, o in model.field_offsets.items()})}")
    cams = [f for f in obs if obs[f].ndim == 5]
    print(f"cameras   {len(cams)} x {tuple(obs[cams[0]].shape[-2:]) if cams else '-'}")

    with torch.inference_mode():
        # reported on its own: with --compile this call is mostly Dynamo
        first = time.perf_counter()
        with amp():
            model.predict(obs)
        first = time.perf_counter() - first

        def predict():
            with amp():
                model.predict(obs)

        times = time_it(predict, args.iters)

        with amp():
            clean, sources, cond, hidden = model._condition(obs)
            seeds = model.build_flow_seeds(
                obs, {**clean, **sources, **cond, **hidden}, training=False
            )

        def encode():
            with amp():
                model._condition(obs)

        def rollout():
            with amp():
                model.predictor.rollout(
                    clean, sources, cond=cond, num_steps=model.num_flow_steps,
                    cfg_scale=model.cfg_scale, cfg_drop=model.cfg_drop, x0=seeds,
                )

        def encode_one_camera():
            with amp():
                model.encoders["vjepa"](obs[cams[0]])

        # same ViT, both camera windows in one batch. Fed at the encoder's own crop_size
        # so the cameras' differing native resolutions stay out of the comparison.
        crop = model.encoders["vjepa"].crop_size
        steps = obs[cams[0]].shape[1]
        solo = torch.randint(0, 256, (1, steps, 3, *crop), dtype=torch.uint8, device=args.device)
        both = solo.expand(len(cams), *solo.shape[1:]).contiguous()

        def encode_solo():
            with amp():
                model.encoders["vjepa"](solo)

        def encode_batched():
            with amp():
                model.encoders["vjepa"](both)

        encode_s = time_it(encode, args.iters)
        rollout_s = time_it(rollout, args.iters)
        vjepa_s = time_it(encode_one_camera, args.iters) if cams else [0.0]
        solo_s = time_it(encode_solo, args.iters) if cams else [0.0]
        batched_s = time_it(encode_batched, args.iters) if cams else [0.0]

    median = statistics.median(times)
    chunk_s = model.execute_len / float(cfg.algorithm.action_fps)

    print(f"\npredict() over {args.iters} runs")
    print(f"  first   {first:.3f}s   ({'compile + warmup' if compiled else 'warmup'}, excluded below)")
    print(f"  min     {times[0]:.3f}s")
    print(f"  median  {median:.3f}s")
    print(f"  max     {times[-1]:.3f}s")
    print(f"  rate    {1 / median:.2f} Hz")
    print(f"  budget  {chunk_s:.2f}s of actions per replan "
          f"({'fits' if median < chunk_s else f'{median / chunk_s:.1f}x too slow'})")

    print("\nwhere it goes (median)")
    print(f"  V-JEPA, one camera window        {statistics.median(vjepa_s):.3f}s")
    print(f"  all encoders (_condition)        {statistics.median(encode_s):.3f}s"
          f"   ({statistics.median(encode_s) / median:.0%})")
    print(f"  flow rollout ({model.num_flow_steps} trunk passes)     "
          f"{statistics.median(rollout_s):.3f}s   ({statistics.median(rollout_s) / median:.0%})")

    if cams:
        solo, batched = statistics.median(solo_s), statistics.median(batched_s)
        n = len(cams)
        print(f"\nbatching the {n} cameras into one V-JEPA call (at {model.encoders['vjepa'].crop_size})")
        print(f"  {n} calls of batch 1              {n * solo:.3f}s")
        print(f"  1 call of batch {n}               {batched:.3f}s")
        print(f"  saving                           {n * solo - batched:+.3f}s"
              f"   ({1 - batched / (n * solo):.0%} of the encoder)")


if __name__ == "__main__":
    main()
