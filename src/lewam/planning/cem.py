"""Cross-entropy-method planner, after external/vjepa2/notebooks/utils/mpc_utils.py.

Meta's version samples 4 dims (xyz + gripper) for one environment; this one samples an
arbitrary (horizon, dim) block for a batch of environments at once. Same update: Gaussian
proposals, top-k elites, mean/std refit blended with the previous iterate through a
momentum term, the final mean returned as the plan.
"""

import torch

DEFAULTS = dict(
    num_samples=128,
    n_steps=20,
    topk=16,
    init_std=0.1,
    momentum_mean=0.25,
    momentum_std=0.75,
    clip=3.0,
)


def cem(cost_fn, init_mean, cfg=None, generator=None):
    """`init_mean` (N, H, D) -> plan (N, H, D).

    `cost_fn` maps candidates (N*S, H, D) to costs (N*S,), lower is better; it is called
    once per iteration with every environment's samples stacked. Everything is in the
    units `init_mean` is in; `clip` bounds every coordinate to [-clip, clip].
    """
    cfg = {**DEFAULTS, **dict(cfg or {})}
    S, K = int(cfg["num_samples"]), int(cfg["topk"])
    if not 1 <= K <= S:
        raise ValueError(f"topk={K} must be within [1, num_samples={S}]")
    clip = float(cfg["clip"])
    mm, ms = float(cfg["momentum_mean"]), float(cfg["momentum_std"])
    mean = init_mean.detach().clone().clamp(-clip, clip)
    std = torch.full_like(mean, float(cfg["init_std"]))
    N, H, D = mean.shape
    for _ in range(int(cfg["n_steps"])):
        eps = torch.randn(N, S, H, D, device=mean.device, dtype=mean.dtype, generator=generator)
        cand = (mean[:, None] + std[:, None] * eps).clamp(-clip, clip)
        cost = cost_fn(cand.flatten(0, 1)).view(N, S)
        idx = cost.topk(K, dim=1, largest=False).indices              # (N, K)
        elite = cand.gather(1, idx[:, :, None, None].expand(N, K, H, D))
        mean = elite.mean(dim=1) * (1 - mm) + mean * mm
        if K > 1:
            std = elite.std(dim=1) * (1 - ms) + std * ms
    return mean.clamp(-clip, clip)
