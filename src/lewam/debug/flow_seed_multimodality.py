"""Does an A2A seed cost the policy its multimodality? The A2A paper's 2D navigation task.

An agent must get from the start (*) to the target (+) past an upper obstacle and through a
gap in a lower wall. Going left or right of the upper obstacle are both optimal, so the
demonstrations are multimodal for one and the same observation.

Every arm trains the same flow-matching objective on the same data; only `predict.action.source`
(equivalently source_std below) differs, plus which network does the predicting -- pick with
--model:

    transformer   the real GenericViTPredictor on a config built in this file, tiny but
                  structurally identical to configs/algorithm/flow_common.yaml: a state
                  encoder over the current step and one flow stream over the next action
                  chunk -- see build_cfg's docstring for what conditioning on observations
                  alone costs on this task
    mlp           a plain adaLN-conditioned residual MLP (MLPFlow below), no attention, no
                  config system -- the state is concatenated as input alongside the noised
                  flow workspace, timestep enters through the same adaLN-Zero primitives
                  (TimestepEmbedder/InputLayer/FinalLayer) the transformer's DiT trunk uses
    transformer_dz  the transformer arm with DreamZero's token in/out in place of the adaLN
                  projections -- the flow stays in raw (x, y) space and the loss is
                  identical, only how the chunk enters and leaves the trunk moves, so this
                  arm isolates their I/O structure from their seed. See build_cfg_dz
    transformer_flat  the transformer arm with the whole chunk as ONE token: the CHUNK raw
                  (x, y) rows are concatenated into a single 2*CHUNK vector, linearly
                  projected into the trunk by the same InputLayer, and read back out by the
                  same FinalLayer. Flow space is still raw action space and the objective is
                  byte-for-byte the transformer arm's, so this isolates per-step tokenization
                  -- see build_cfg_flat for why that is the suspect on an a2a seed
    transformer_flat_dz  transformer_flat with DreamZero's token in/out, which is the
                  combination the b601 arms run. See build_cfg_flat_dz
    transformer_ae  the transformer arm, but the flow stream is the action chunk compressed
                  through a chunk_mlp encoder into one latent token instead of raw (x, y)
                  space, decoded back by its mirror -- see build_cfg_ae's docstring. Trains
                  with two extra loss terms: `ae_recon` (grounds the codec against
                  collapse) and `integration` (Euler-solves the codec-latent flow WITH
                  gradients for AE_INT_STEPS steps and scores the raw-space endpoint,
                  supervising the multi-step solve rollout actually runs, not just a
                  velocity at one random t)

All three read a `source_std` the same way conceptually --

    noise      source_std is None    x0 ~ N(0, I), standard flow matching
    a2a        source_std 0.0        x0 IS the previous chunk
    a2a+s      source_std s          x0 is the previous chunk, perturbed

-- and no arm reads the action history as conditioning, matching DreamZero, which
conditions on observations alone. So x0 is now the ONLY route the previous chunk has into
an a2a arm, and the noise arm has no access to it at all: the two no longer differ in where
the same information enters, they differ in whether the model gets it. transformer_ae
differs further in what space the flow stream itself lives in (a codec latent vs raw (x, y)).

That makes the prediction sharp. A2A's x0 is a deterministic function of the conditioning,
so p(chunk | obs) is a point mass and rollouts must commit to one side of the obstacle no
matter what the demonstrations do. Perturbing the seed is the only entropy an A2A arm has,
and the sweep asks how little of it buys the second mode back.

Actions are absolute waypoints rather than deltas, matching b601, whose action column is a
joint-position target tracking the next measured state and moves ~1% of its own magnitude
per step. The action space is not a detail here: in an absolute space x0 nearly IS x1, so
the seed anchors far harder than a chunk of near-zero-mean deltas would, and the reported
transport distance is what makes a noise_std comparable to the one in seed_a2a_raw_noise.

Rollouts start from rest with the start position held as the action history, jittered
START_JITTER across rollouts -- without that jitter "A2A produced one trajectory" would be
true by arithmetic and would show nothing.

    .venv/bin/python src/lewam/debug/flow_seed_multimodality.py
        [--model transformer|mlp|transformer_dz|transformer_flat|transformer_flat_dz|
                 transformer_ae]
        [--block-size K] [--demos N] [--out PREFIX] [--only-arm NAME] [--trace]

Trains one model per arm, on cuda / mps / cpu in that order of preference (--device
overrides); roughly an hour for the whole sweep on CPU (the mlp model is much faster).
Writes <prefix>_data.png and <prefix>_rollouts.png. --only-arm restricts the sweep to one
arm (e.g. `--only-arm a2a`), and --trace additionally dumps that arm's raw rollout
trajectory to <prefix>_<arm>_trace.npy and (mlp only) prints predict()'s per-Euler-substep
|v|/|x| for every rolled-out chunk, for inspecting closed-loop instability directly.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf

from lewam.algorithms import build_algorithm
from lewam.algorithms.layers import (FinalLayer, InputLayer, TimestepEmbedder, bcast_cond,
                                     make_mlp, modulate)

START, TARGET = np.array([0.0, 1.0]), np.array([0.0, -1.0])
# (x0, y0, x1, y1); the upper bar must be rounded, the lower pair leaves a gap at |x| < 0.18
OBSTACLES = [(-0.35, 0.27, 0.35, 0.33),
             (-1.15, -0.33, -0.18, -0.27),
             (0.18, -0.33, 1.15, -0.27)]
GAP_Y, CROSS_Y = -0.30, 0.30
STEP = 0.04           # arclength between consecutive demo positions, so speed is constant
START_JITTER = 0.03
WIDE_CUT = 0.75       # |x| at the obstacle separating a narrow detour from a wide one

CHUNK, STATE_STEPS = 8, 1   # one state step: DreamZero's single state token, see build_cfg
START_HOLD = 3 * CHUNK   # frames of noisy home-hold inserted after the exact front pad
START_HOLD_JITTER = 0.02
END_HOLD = 3 * CHUNK     # frames of noisy goal-hold appended before the exact-repeat back pad
END_JITTER = 0.02        # iid per-frame noise during either hold, so home/goal windows
                         # aren't an exact x0->x1 identity the closed a2a loop can't correct
STATE_DIM, ACTION_DIM = 2, 2
# spelled out rather than f"-{STATE_STEPS - 1}..0" so raising STATE_STEPS stays one edit
STATE_RAW_INDEX = "0" if STATE_STEPS == 1 else f"-{STATE_STEPS - 1}..0"
# width, FFN ratio and dropout are MLPFlow's, so --model reads as a comparison of
# structure, not capacity; shrinking the transformer below this broke even the noise arm
TRANSFORMER_DIM, TRANSFORMER_MLP_RATIO, TRANSFORMER_DROPOUT = 128, 4.0, 0.0
DZ_HEAD_DIM = 16        # --model transformer_dz's token_out bottleneck; see build_cfg_dz
AE_INT_STEPS = 4        # --model transformer_ae's integration loss: Euler steps solved
                        # WITH gradients per training step (num_flow_steps is 10, used only
                        # at eval/rollout -- this is deliberately fewer, to bound the cost)
FAMILIES = ["left", "right", "left wide", "right wide"]
# roughly doubling, to bracket both the A2A paper's 0.02 and seed_a2a_raw_noise's 0.3.
# Read each against the transport distance the run prints, not against the raw action scale
SEED_NOISE = [0.0, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30, 0.60]
ARMS = ([("noise", None)]
        + [(f"a2a+{s:g}" if s else "a2a", s) for s in SEED_NOISE])



def waypoints(family, rng):
    side = -1.0 if family.startswith("left") else 1.0
    start = START + rng.normal(0, START_JITTER, 2)
    if "wide" in family:
        pts = [start, (side * 0.72, 0.88), (side * 0.98, 0.52), (side * 0.96, 0.08),
               (side * 0.58, -0.16), (side * 0.12, -0.27), (0.0, GAP_Y - 0.06), TARGET]
    else:
        pts = [start, (side * 0.48, 0.74), (side * 0.52, 0.40), (side * 0.47, 0.04),
               (side * 0.14, -0.20), (0.0, GAP_Y - 0.06), TARGET]
    pts = np.array(pts, dtype=float)
    pts[1:-1] += rng.normal(0, 0.035, pts[1:-1].shape)
    return pts


def _densify(pts, n):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    u = np.linspace(0.0, t[-1], n)
    return np.stack([np.interp(u, t, pts[:, i]) for i in range(2)], axis=1)


def _smooth(path, width):
    k = np.ones(width) / width
    cols = [np.convolve(np.pad(path[:, i], width, mode="edge"), k, mode="same")[width:-width]
            for i in range(2)]
    return np.stack(cols, axis=1)


def _resample(path, step):
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    return np.stack([np.interp(np.arange(0.0, t[-1], step), t, path[:, i])
                     for i in range(2)], axis=1)


def collides(path):
    dense = _densify(path, len(path) * 4)
    return any(np.any((dense[:, 0] >= x0) & (dense[:, 0] <= x1) &
                      (dense[:, 1] >= y0) & (dense[:, 1] <= y1))
               for x0, y0, x1, y1 in OBSTACLES)


def demo(family, rng):
    """One scripted trajectory, padded at both ends so a window at the start of the run
    carries the all-zero (fully constant, no motion) state and action history a rollout
    begins from -- windows()'s earliest window (t=CHUNK) reads its history from
    act[0:CHUNK] = padded_path[1:CHUNK+1], entirely inside the exact CHUNK-length front
    pad, so it exactly matches rollout()'s true init condition (no noise). START_HOLD
    frames of home position + noise follow that exact pad, before the real trajectory
    starts: later windows (t > CHUNK) sweep into this noisy region, so the network also
    sees near-home-with-noise -> continue-toward-home-then-depart examples, not just the
    single exact fixed point every demo would otherwise share at the home position.

    The tail mirrors this for the same reason: an exact-repeat hold at the goal only ever
    teaches x0 == goal -> x1 == goal, the identity map. A rollout that lands near but not
    exactly on that fixed point (any real closed-loop rollout, since nothing is exact) is
    off the trained manifold, and the network has never seen what to do about it -- which
    is why the a2a arm used to reach the target and then diverge instead of staying put.
    END_HOLD frames of goal position + noise put many slightly-off-goal points into
    training, each still targeting a near-goal chunk, so the learned field damps drift
    back toward the goal instead of being undefined there.
    """
    path = _resample(_smooth(_densify(waypoints(family, rng), 600), 45), STEP)
    start_hold = path[:1] + rng.normal(0, START_HOLD_JITTER, (START_HOLD, 2))
    end_hold = path[-1] + rng.normal(0, END_JITTER, (END_HOLD, 2))
    return np.concatenate([np.repeat(path[:1], CHUNK, 0), start_hold, path, end_hold,
                           np.repeat(path[-1:], CHUNK, 0)])


def build_demos(per_family, rng):
    paths, labels = [], []
    for fam in FAMILIES:
        for _ in range(per_family):
            for _attempt in range(50):
                path = demo(fam, rng)
                if not collides(path):
                    break
            else:
                raise RuntimeError(f"could not script a collision-free '{fam}' demo")
            paths.append(path)
            labels.append(fam)
    return paths, labels


def windows(paths):
    """Every (state window, action window) pair the config's indices can read.

    Rows follow the per-field union the model derives from its own indices: `action` spans
    -CHUNK..CHUNK-1 with history first, `observation.state` spans -(STATE_STEPS-1)..0.
    """
    states, actions = [], []
    for path in paths:
        # absolute waypoints, not deltas: b601's action column is a joint-position target
        # tracking the next measured state, and A2A anchors far harder in that space
        act = path[1:]
        for t in range(max(STATE_STEPS - 1, CHUNK), len(act) - CHUNK + 1):
            states.append(path[t - STATE_STEPS + 1:t + 1])
            actions.append(act[t - CHUNK:t + CHUNK])
    return np.stack(states), np.stack(actions)



def build_cfg(noise_std, block_size=None):
    """flow_common.yaml's structure at a quarter of the width, with the vision streams
    dropped and MLPFlow's capacity (see TRANSFORMER_DIM).

    DreamZero conditions on observations alone -- no action history anywhere, and one
    state token per chunk rather than a window -- so `action_context` is `via: null` here
    exactly as in flow_common.yaml (encoded for nothing, surviving only to carry
    `role: action`), and STATE_STEPS is 1.

    That is a hostile setting for this task and deliberately so. An earlier version of
    this function dropped action_context and even the *noise* arm fell to 0% success
    despite a fine training loss, while restoring it brought the arm back to 100%; the
    trunk could not fit the data without it. With STATE_STEPS at 1 the observation is a
    single position, so which way the agent is travelling is not observable at all and
    the four demo families are indistinguishable at most points on the map. Raise
    STATE_STEPS to recover velocity, and give action_context a `via` back to recover the
    history channel -- both are one edit, and the gap between those runs and this one is
    what conditioning on observations alone actually costs.

    `block_size` None (the default, matching flow_common.yaml) means `causal: false`:
    attention is unmasked and the argument does not apply. Passing one turns the time mask
    on at that granularity, which matters far more with an A2A seed than a noise one. At 1
    the chunk is strictly causal per action step, so step k attends seed rows <= k only:
    step 0's whole view of the history is its own x0, the OLDEST row in the window. A noise
    seed carries nothing to be starved of and does not care; an A2A seed does. At CHUNK the
    whole chunk is one block attending itself fully -- for this config, indistinguishable
    from unmasked, since the only other token is the state at time 0.
    """
    source = None if noise_std is None else {
        "from": "action", "index": f"-{CHUNK}..-1", "noise_std": noise_std,
    }
    action = {"from": "action", "type": "flow", "dim": 2, "index": f"0..{CHUNK - 1}",
              "fps": 1.0, "block_size": block_size or 1}
    if source is not None:
        action["source"] = source
    dim = TRANSFORMER_DIM
    return OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": {
            "model_dim": dim, "depth": 3, "num_heads": 4,
            "mlp_ratio": TRANSFORMER_MLP_RATIO, "dropout": TRANSFORMER_DROPOUT,
            "causal": block_size is not None,
            "rope": {"time": {"share": 0.75, "period": "auto"},
                     "seq": {"share": 0.25, "period": "auto"}},
        },
        "num_flow_steps": 10,
        "encoders": {"state_enc": {
            "type": "chunk_mlp", "in_dim": 2, "in_steps": STATE_STEPS, "out_dim": dim,
            "out_steps": 1, "hidden_dim": 2 * dim, "depth": 2, "norm": "layer",
        }},
        "conditioning": {
            "state_context": {
                "from": "observation.state", "encoder": "state_enc", "dim": dim,
                "index": "0", "raw_index": STATE_RAW_INDEX, "fps": 1.0,
                "grid": [1, 1],
            },
            "action_context": {
                "from": "action", "role": "action", "via": None, "dim": 2,
                "index": f"-{CHUNK}..-1", "fps": 1.0,
            },
        },
        "predict": {"action": action},
        "losses": [{"type": "flow", "stream": "action", "weight": 1.0}],
    })


def build_cfg_dz(noise_std, block_size=None):
    """build_cfg's transformer, with DreamZero's action encoder/decoder replacing the
    adaLN InputLayer/FinalLayer on the predicted stream.

    `token_in` is their MultiEmbodimentActionEncoder: a linear lift to model_dim, the
    timestep embedding concatenated onto it, then an MLP back down (their W2/W3 are
    2w -> w -> w, hence mlp_dim = model_dim), so t enters here as well as through adaLN.
    `token_out` is their action decoder, an MLP through a narrow bottleneck with no norm and
    no modulation, though its output projection is zero-initialized here as FinalLayer's is,
    so both arms start at zero velocity and the comparison is of structure rather than of
    init. Their 64 does not carry over by either ratio it could (2x the action dim, or
    model_dim/80); DZ_HEAD_DIM is a real bottleneck at this width instead.

    Nothing else changes: raw (x, y) flow space, same losses, same seeds, so the sweep
    reads against --model transformer arm for arm.
    """
    cfg = build_cfg(noise_std, block_size)
    cfg.predict.action.token_in = {"type": "mlp", "mlp_dim": TRANSFORMER_DIM}
    cfg.predict.action.token_out = {"type": "mlp", "mlp_dim": DZ_HEAD_DIM}
    return cfg


def build_cfg_flat(noise_std, block_size=None):
    """build_cfg's transformer with the chunk as ONE token instead of CHUNK of them, still
    flowing in raw action space: a parameter-free `reshape` concatenates the CHUNK (x, y)
    rows into one 2*CHUNK vector (its mirror splits the ODE endpoint back into rows), and
    the trunk's ordinary InputLayer/FinalLayer are the only projection in or out. Same
    conditioning, same flow loss, same seeds as --model transformer; nothing but the token
    layout of the predicted stream moves.

    So this is transformer_ae's single-token shape without its learned codec, and it exists
    to separate the two: an exact reshape has no latent to collapse and needs no ae_recon,
    sigreg or integration term propping it up, so whatever it recovers over --model
    transformer is attributable to tokenization alone.

    What tokenization costs an A2A arm, and only an A2A arm: actions here are absolute
    waypoints STEP apart, so the CHUNK seed rows are near-identical vectors and everything
    the seed adds over a noise seed lives in the few-percent differences BETWEEN them. Per
    step, each row is a 2-D token lifted by InputLayer and then RMSNormed on its own (as it
    is again in every DiTBlock branch), which is close to keeping only that waypoint's
    direction, and softmax attention over CHUNK near-identical tokens returns roughly their
    mean -- the common mode survives, the differential signal is what gets attenuated. As one
    token the differences are just separate coordinates of one vector, and "advance the chunk
    by CHUNK steps" is a single weight matrix, which is exactly what MLPFlow gets for free by
    flattening. A noise seed is unaffected either way: x0 ~ N(0, I) carries nothing to lose,
    and the rows are maximally distinct to begin with.

    `block_size` is accepted for a uniform call site but is inert: one token is its own block.
    """
    cfg = build_cfg(noise_std, block_size)
    flat_dim = CHUNK * ACTION_DIM
    cfg.encoders.act_flat = {"type": "reshape", "in_dim": ACTION_DIM, "in_steps": CHUNK,
                             "out_dim": flat_dim, "out_steps": 1}
    cfg.encoders.act_unflat = {"type": "reshape", "in_dim": flat_dim, "in_steps": 1,
                               "out_dim": ACTION_DIM, "out_steps": CHUNK}
    action = {"from": "action", "type": "flow", "encoder": "act_flat", "decoder": "act_unflat",
              "dim": flat_dim, "index": "0", "raw_index": f"0..{CHUNK - 1}",
              "chunk_len": CHUNK, "fps": 1.0}
    if noise_std is not None:
        action["source"] = {"from": "action", "index": f"-{CHUNK}..-1", "encoder": "act_flat",
                            "noise_std": noise_std}
    cfg.predict = {"action": action}
    # as in build_cfg_ae: two tokens, state and chunk, both at time 0, so `time` has no
    # spread for an automatic period to fit
    cfg.model.rope = {"seq": {"share": 1.0, "period": "auto"}}
    return cfg


def build_cfg_flat_dz(noise_std, block_size=None):
    """build_cfg_flat with DreamZero's token in/out, which is what the b601 arms now run:
    one raw-space chunk token, entering and leaving through their MLP pair instead of the
    adaLN projections. build_cfg_dz's overlay on build_cfg_flat's layout, so it reads against
    --model transformer_flat arm for arm and the only variable is their I/O.

    `token_out` takes the trunk width, not DZ_HEAD_DIM: MLPFinalLayer's `mlp_dim` bounds the
    RANK of what it emits, and this stream emits the whole chunk at once. DreamZero's own 64
    expands into a single `action_dim` row and constrains nothing there, but against
    CHUNK * ACTION_DIM it would confine every predicted chunk to a 64-dim subspace -- an
    accident of porting the number rather than their design.
    """
    cfg = build_cfg_flat(noise_std, block_size)
    cfg.predict.action.token_in = {"type": "mlp", "mlp_dim": TRANSFORMER_DIM}
    cfg.predict.action.token_out = {"type": "mlp", "mlp_dim": TRANSFORMER_DIM}
    return cfg


def build_cfg_ae(noise_std, block_size=None):
    """build_cfg's transformer, but the predicted action chunk flows in a chunk_mlp codec
    latent instead of raw (x, y) space: `act_enc` compresses the CHUNK raw rows the flow
    stream targets (and, as an a2a seed, the CHUNK raw rows of history) into one latent
    token; `act_dec` is its mirror, applied to the ODE endpoint so predict()/rollout still
    hand back raw chunks. Conditioning is build_cfg's, action history included in nothing --
    only the predicted/seeded stream's own representation changes, matching
    the "flowing the action chunk in a learned codec latent" recipe (`configs/algorithm/wam_a2a_int_b601.yaml` before it was retired from the sweep,
    recoverable at `git show <commit>^:configs/algorithm/wam_a2a_int_b601.yaml`).

    ae_recon grounds the codec against collapse (nothing here stop-grads it); sigreg
    additionally regularizes the latent toward isotropic Gaussian, so the noise arm's
    x0 ~ N(0, I) seed actually lands near the codec's real data manifold instead of
    wherever an unconstrained encoder happened to put it; integration Euler-solves the
    codec-latent flow WITH gradients for `AE_INT_STEPS` steps and scores the raw-space
    endpoint, supervising the same multi-step solve rollout actually runs instead of only
    a velocity at a random t.

    `block_size` is accepted for a uniform call site but is inert here: the codec leaves
    the stream one token, which is its own block whatever the value.
    """
    dim = TRANSFORMER_DIM
    action = {
        "from": "action", "type": "flow", "encoder": "act_enc", "decoder": "act_dec",
        "dim": dim, "index": "0", "raw_index": f"0..{CHUNK - 1}", "chunk_len": CHUNK,
        "fps": 1.0,
    }
    if noise_std is not None:
        action["source"] = {"from": "action", "index": f"-{CHUNK}..-1", "encoder": "act_enc",
                            "noise_std": noise_std}
    return OmegaConf.create({
        "name": "generic_vit_predictor",
        "model": {
            "model_dim": dim, "depth": 3, "num_heads": 4,
            "mlp_ratio": TRANSFORMER_MLP_RATIO, "dropout": TRANSFORMER_DROPOUT,
            # no `time` axis: the codec leaves this arm two tokens, the state and the one
            # latent, and both sit at time 0, so an automatic period has nothing to fit
            "rope": {"seq": {"share": 1.0, "period": "auto"}},
        },
        "num_flow_steps": 10,
        "encoders": {
            "state_enc": {
                "type": "chunk_mlp", "in_dim": 2, "in_steps": STATE_STEPS, "out_dim": dim,
                "out_steps": 1, "hidden_dim": 2 * dim, "depth": 2, "norm": "layer",
            },
            "act_enc": {
                "type": "chunk_mlp", "in_dim": ACTION_DIM, "in_steps": CHUNK, "out_dim": dim,
                "out_steps": 1, "hidden_dim": 2 * dim, "depth": 2,
            },
            "act_dec": {
                "type": "chunk_mlp", "in_dim": dim, "in_steps": 1, "out_dim": ACTION_DIM,
                "out_steps": CHUNK, "hidden_dim": 2 * dim, "depth": 2,
            },
        },
        "conditioning": {
            "state_context": {
                "from": "observation.state", "encoder": "state_enc", "dim": dim,
                "index": "0", "raw_index": STATE_RAW_INDEX, "fps": 1.0,
                "grid": [1, 1],
            },
            "action_context": {
                "from": "action", "role": "action", "via": None, "dim": 2,
                "index": f"-{CHUNK}..-1", "fps": 1.0,
            },
        },
        "predict": {"action": action},
        "losses": [
            {"type": "flow", "stream": "action", "weight": 1.0},
            {"type": "ae_recon", "stream": "action", "decoder": "act_dec", "weight": 1.0,
             "norm": "l1"},
            {"type": "sigreg", "streams": ["action"], "weight": 0.1},
        ] + ([{"type": "integration", "stream": "action", "decoder": "act_dec", "weight": 1.0,
               "norm": "l1", "num_steps": AE_INT_STEPS}] if noise_std is not None else []),
    })


class AdaLNMLPBlock(nn.Module):
    """Residual SwiGLU MLP, adaLN-Zero modulated by cond -- the MLP half of a DiT block
    (lewam.algorithms.block.DiTBlock) with the attention half dropped, since this trunk
    has no tokens to attend over. Zero-initialized so the block is the identity at init."""

    def __init__(self, hidden, mlp_ratio=4.0, cond_dim=None):
        super().__init__()
        self.norm = nn.RMSNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.mlp = make_mlp(hidden, int(hidden * mlp_ratio), hidden)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(cond_dim or hidden, 3 * hidden, bias=True))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, cond):
        shift, scale, gate = self.adaLN_modulation(cond).chunk(3, dim=-1)
        h = modulate(self.norm(x), shift, scale)
        return x + bcast_cond(gate) * self.mlp(h)


class MLPFlow(nn.Module):
    """The --model mlp arm: flow matching over a flattened action chunk, no attention, no
    config system, the network is a stack of AdaLNMLPBlocks.

    Conditioning mirrors the transformer arm's structure: the state alone, concatenated as
    a plain input alongside the noised flow workspace `xt`, which every ODE step also sees.
    It carries no action history -- the MLP's analogue of `action_context` was dropped with
    it, so here too the only action information an a2a arm has is x0. x0 (xt's start) is
    either standard Gaussian noise (source_std is None, the "noise" arm) or the previous
    action chunk perturbed by source_std (the "a2a[+s]" arms). Time enters through
    adaLN-Zero (TimestepEmbedder + InputLayer/AdaLNMLPBlock/FinalLayer, the same primitives
    the transformer arm's DiT trunk uses), not concatenation, so the network can't just
    learn a fixed offset for t.
    """

    def __init__(self, source_std=None, hidden=128, depth=3):
        super().__init__()
        self.source_std = source_std
        self.chunk, self.action_dim = CHUNK, ACTION_DIM
        act_dim = CHUNK * ACTION_DIM
        context_dim = STATE_STEPS * STATE_DIM

        self.t_embedder = TimestepEmbedder(hidden)
        self.in_layer = InputLayer(context_dim + act_dim, hidden, cond_dim=hidden)
        self.blocks = nn.ModuleList([AdaLNMLPBlock(hidden, cond_dim=hidden) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden, act_dim, cond_dim=hidden)

    def _seed(self, hist_flat, x1):
        if self.source_std is None:
            return torch.randn_like(x1)
        return hist_flat + self.source_std * torch.randn_like(x1)

    def _velocity(self, context, xt, t):
        t_emb = self.t_embedder(t.squeeze(-1))
        # InputLayer/AdaLNMLPBlock/FinalLayer modulate (B, N, D) tokens; we have one
        # "token" per example, so add and drop that axis around the trunk.
        h = self.in_layer(torch.cat([context, xt], dim=-1).unsqueeze(1), t_emb)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.final_layer(h, t_emb).squeeze(1)

    def loss(self, batch):
        state, actions = batch["observation.state"], batch["action"]
        b = state.shape[0]
        hist, target = actions[:, :self.chunk], actions[:, self.chunk:]
        hist_flat, x1 = hist.reshape(b, -1), target.reshape(b, -1)
        context = state.reshape(b, -1)
        x0 = self._seed(hist_flat, x1)
        t = torch.rand(b, 1, device=x1.device)
        xt = (1 - t) * x0 + t * x1
        v_pred = self._velocity(context, xt, t)
        return {"loss": ((v_pred - (x1 - x0)) ** 2).mean()}

    @torch.no_grad()
    def predict(self, obs, num_steps=10, trace=False):
        """trace=True prints the ODE's per-substep velocity/state norms, to see whether a
        rollout is settling into a smooth trajectory or bouncing around an unstable point
        in the learned field (e.g. the a2a arm with no seed noise, off the data manifold)."""
        state, hist = obs["observation.state"], obs["action"]
        b = state.shape[0]
        hist_flat = hist.reshape(b, -1)
        context = state.reshape(b, -1)
        x = self._seed(hist_flat, hist_flat)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((b, 1), i * dt, device=state.device)
            v = self._velocity(context, x, t)
            if trace:
                print(f"    substep {i:2d}: |v| mean {v.norm(dim=-1).mean().item():7.3f}  "
                      f"max {v.norm(dim=-1).max().item():7.3f}   "
                      f"|x| mean {x.norm(dim=-1).mean().item():7.3f}  "
                      f"max {x.norm(dim=-1).max().item():7.3f}")
            x = x + v * dt
        return {"action": x.reshape(b, self.chunk, self.action_dim)}


class Scaler:
    """Per-dim normalization. Actions are absolute positions, so they carry a mean."""

    def __init__(self, states, actions):
        self.s_mean, self.s_std = states.mean((0, 1)), states.std((0, 1)) + 1e-6
        self.a_mean, self.a_std = actions.mean((0, 1)), actions.std((0, 1)) + 1e-6

    def state(self, p):
        return (p - self.s_mean) / self.s_std

    def action(self, a):
        return (a - self.a_mean) / self.a_std

    def unaction(self, a):
        return a * self.a_std + self.a_mean


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(model_type, source_std, block_size=None, compile_model=False):
    if model_type == "mlp":
        # no tokens, so nothing for block_size to make causal: the flattened chunk is
        # visible to every layer at once, which is the point of comparing against it
        model = MLPFlow(source_std)
    elif model_type == "transformer_ae":
        model = build_algorithm(build_cfg_ae(source_std, block_size))
    elif model_type == "transformer_flat":
        model = build_algorithm(build_cfg_flat(source_std, block_size))
    elif model_type == "transformer_flat_dz":
        model = build_algorithm(build_cfg_flat_dz(source_std, block_size))
    elif model_type == "transformer_dz":
        model = build_algorithm(build_cfg_dz(source_std, block_size))
    else:
        model = build_algorithm(build_cfg(source_std, block_size))
    if compile_model:
        # dispatch/kernel-launch overhead dominates wall time for a model this small;
        # compiling fuses each block's many tiny ops into fewer launches. The script calls
        # model.loss(...)/model.predict(...), not model(...), so torch.compile(model)
        # (which only hooks forward()) would silently compile nothing -- compile the bound
        # methods actually on the hot path instead. Falls back to eager per-method on any
        # backend that can't handle it, instead of hard-failing a long run over a speedup.
        for name in ("loss", "predict"):
            if hasattr(model, name):
                try:
                    setattr(model, name, torch.compile(getattr(model, name)))
                except Exception as e:
                    print(f"torch.compile({name}) failed ({e}); continuing uncompiled")
    return model


def train(model_type, source_std, states, actions, steps, batch, lr, log_every,
         block_size=None, compile_model=False):
    import time
    torch.manual_seed(0)
    model = build_model(model_type, source_std, block_size, compile_model).to(states.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    model.train()
    n, trace = states.shape[0], []
    t0 = time.perf_counter()
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=states.device)
        loss = model.loss({"observation.state": states[idx], "action": actions[idx]})["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % log_every == 0:
            trace.append(loss.item())
            rate = (step + 1) / (time.perf_counter() - t0)
            print(f"\rstep {step}/{steps}  loss {loss.item():.4f}  {rate:.1f} it/s\033[K",
                  end="", flush=True)
    print(f"\r\033[K", end="", flush=True)
    return model, trace



@torch.no_grad()
def rollout(model, scaler, n_rollouts, max_chunks, rng, trace=False, repeat_last=False,
           exec_noise=0.0):
    """Closed loop from rest, all rollouts stepped as one batch.

    The action history starts as the start position held (which the demos' front padding
    makes in-distribution) and the start position is jittered, so any spread across
    rollouts is the policy's own.

    trace=True prints predict()'s per-substep ODE diagnostics for every chunk (mlp arm
    only -- the transformer's predict() doesn't expose per-step internals), to inspect
    whether a rollout is compounding instability closed-loop: each chunk's seed is the
    model's own previous output, so with no seed noise there's nothing to correct drift.

    repeat_last=True feeds the *next* step's action history as CHUNK copies of just the
    most recently predicted position, instead of the full CHUNK-length chunk the model
    just predicted. Normally hist_a carries the model's own predicted trajectory shape
    forward as next step's conditioning/a2a seed -- any artifact in that shape re-enters
    directly. repeat_last resets it to a plain "holding position" history every step (the
    same shape rollouts already start from at c=0), isolating whether shape-carryover is
    itself a source of the closed-loop instability the a2a arm shows.

    exec_noise adds iid Gaussian noise (this std, in raw action units) to every predicted
    position before it's recorded and fed back as the next step's history/state -- what a
    real robot's tracking error/proprioception noise does to a commanded action, which this
    sim's exact float-for-float feedback otherwise never has. A locally-stable closed-loop
    fixed point (e.g. a2a's home attractor) that traps a noiseless rollout forever can't
    trap one that gets a fresh perturbation like this every step.
    """
    model.eval()
    device = next(model.parameters()).device
    supports_trace = hasattr(model, "_velocity")  # only MLPFlow exposes step-level internals
    pos = START + rng.normal(0, START_JITTER, (n_rollouts, 2))
    hist_p = np.repeat(pos[:, None], STATE_STEPS, axis=1)
    hist_a = np.repeat(pos[:, None], CHUNK, axis=1)
    paths = [pos.copy()]
    for c in range(max_chunks):
        obs = {
            "observation.state": torch.tensor(scaler.state(hist_p), dtype=torch.float32).to(device),
            "action": torch.tensor(scaler.action(hist_a), dtype=torch.float32).to(device),
        }
        if trace:
            print(f"  chunk {c:2d}")
        if supports_trace:
            action = model.predict(obs, trace=trace)["action"]
        else:
            action = model.predict(obs)["action"]
        chunk = scaler.unaction(action.cpu().numpy())
        if exec_noise:
            chunk = chunk + rng.normal(0, exec_noise, chunk.shape)
        for k in range(CHUNK):
            pos = chunk[:, k]
            paths.append(pos.copy())
        hist_a = np.repeat(chunk[:, -1:], CHUNK, axis=1) if repeat_last else chunk
        hist_p = np.stack(paths[-STATE_STEPS:], axis=1)
    return np.stack(paths, axis=1)


def classify(paths):
    """Per rollout: which side of the upper obstacle it passed, how wide, and whether it
    finished. A rollout that never crosses CROSS_Y has no mode to report."""
    out = []
    for path in paths:
        below = np.nonzero(path[:, 1] <= CROSS_Y)[0]
        x = path[below[0], 0] if len(below) else np.nan
        out.append({
            "x": x,
            "side": "-" if np.isnan(x) else ("left" if x < 0 else "right"),
            "wide": abs(x) >= WIDE_CUT,
            "ok": (not collides(path)
                   and np.linalg.norm(path - TARGET, axis=1).min() < 0.15),
        })
    return out


def summarize(name, modes):
    sides = [m["side"] for m in modes]
    left = sides.count("left") / len(sides)
    wide = np.mean([m["wide"] for m in modes if m["side"] != "-"] or [np.nan])
    ok = np.mean([m["ok"] for m in modes])
    spread = np.nanstd([m["x"] for m in modes])
    return {"arm": name, "left": left, "right": sides.count("right") / len(sides),
            "wide": wide, "success": ok, "spread": spread}



def draw_scene(ax):
    for x0, y0, x1, y1 in OBSTACLES:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="white",
                               edgecolor="0.35", hatch="////", lw=1.0, zorder=3))
    ax.plot(*START, marker="*", ms=16, color="crimson", zorder=5)
    ax.plot(*TARGET, marker="P", ms=11, color="crimson", zorder=5)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_data(paths, labels, out):
    fig, ax = plt.subplots(figsize=(4.2, 4.4))
    colours = dict(zip(FAMILIES, ["tab:blue", "tab:orange", "tab:green", "tab:red"]))
    for path, lab in zip(paths, labels):
        ax.plot(path[:, 0], path[:, 1], color=colours[lab], lw=0.8, alpha=0.35, zorder=2)
    for lab in FAMILIES:
        ax.plot([], [], color=colours[lab], lw=1.6, label=lab)
    draw_scene(ax)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"scripted demonstrations ({len(paths)})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_rollouts(results, out):
    ncols = min(5, len(results))
    nrows = -(-len(results) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.9 * nrows), squeeze=False)
    flat = axes.ravel()
    for ax, (name, paths, modes, stats) in zip(flat, results):
        for path, mode in zip(paths, modes):
            ax.plot(path[:, 0], path[:, 1], lw=1.0, alpha=0.55, zorder=2,
                    color="tab:blue" if mode["side"] == "left" else "tab:green")
        draw_scene(ax)
        ax.set_title(f"{name}\nleft {stats['left']:.0%} / right {stats['right']:.0%}   "
                     f"success {stats['success']:.0%}", fontsize=9)
    for ax in flat[len(results):]:
        ax.axis("off")
    fig.suptitle("rollouts from rest, action history held at the start -- the seed is the "
                 "only difference between these policies", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.06 / nrows))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",
                    choices=["transformer", "mlp", "transformer_dz", "transformer_flat",
                             "transformer_flat_dz", "transformer_ae"],
                    default="transformer",
                    help="transformer = real GenericViTPredictor; mlp = plain adaLN MLP, "
                         "no attention, no config system; transformer_dz = the transformer "
                         "with DreamZero's token in/out, still raw action space; "
                         "transformer_flat = the transformer with the whole chunk as one "
                         "concatenated token, still raw action space and no codec; "
                         "transformer_flat_dz = that layout with DreamZero's token in/out, "
                         "which is what the b601 arms run; "
                         "transformer_ae = GenericViTPredictor "
                         "with the action chunk flowing in a chunk_mlp codec latent (ae_recon "
                         "+ integration losses) instead of raw space")
    p.add_argument("--demos", type=int, default=25, help="per family, 4 families")
    p.add_argument("--rollouts", type=int, default=40)
    p.add_argument("--train-steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-chunks", type=int, default=22)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block-size", type=int, default=None,
                    help=f"turn the time mask ON at this granularity, in action steps. "
                         f"Omitted, attention is unmasked, as flow_common.yaml now is. "
                         f"1 is strictly causal per step, so step k sees seed rows <= k "
                         f"and step 0 sees only the oldest row of the history; {CHUNK} "
                         f"(= CHUNK) makes the chunk one block, which for this config is "
                         f"the same thing as unmasked. Inert for --model mlp, "
                         f"transformer_flat[_dz] and transformer_ae, one action token each")
    p.add_argument("--device", default="auto", help="auto picks cuda, then mps, then cpu")
    p.add_argument("--out", default="/tmp/a2a_nav")
    p.add_argument("--only-arm", default=None,
                    help="run just this arm from ARMS (e.g. --only-arm a2a) instead of the "
                         "full sweep, for fast iteration on one arm")
    p.add_argument("--trace", action="store_true",
                    help="dump the raw rollout trajectory to <out>_<arm>_trace.npy, and "
                         "(mlp only) print predict()'s per-Euler-substep |v|/|x| for every "
                         "rolled-out chunk")
    p.add_argument("--repeat-last", action="store_true",
                    help="feed each step's action history as a repeat of just the most "
                         "recently predicted position, instead of the full predicted "
                         "chunk -- so the model's own predicted trajectory shape never "
                         "carries forward as next step's conditioning/a2a seed")
    p.add_argument("--exec-noise", type=float, default=0.0,
                    help="std of iid Gaussian noise added to every predicted position "
                         "before it's recorded/fed back as history, simulating real "
                         "execution/proprioception noise a sim's exact feedback lacks")
    p.add_argument("--compile", action="store_true",
                    help="torch.compile model.loss/predict -- dispatch overhead dominates "
                         "wall time for a model this small, so this can meaningfully speed "
                         "up training/rollout; falls back to eager per-method if the "
                         "backend can't handle it")
    args = p.parse_args()
    device = pick_device(args.device)
    if device.type == "cpu":
        # a 64-wide trunk over 17 tokens (mlp: a two-hidden-layer MLP over 24 dims) is
        # small enough that one thread per core costs far more in scheduler contention
        # than it wins
        torch.set_num_threads(8)

    rng = np.random.default_rng(args.seed)
    paths, labels = build_demos(args.demos, rng)
    states, actions = windows(paths)
    scaler = Scaler(states, actions)
    ts = torch.tensor(scaler.state(states), dtype=torch.float32).to(device)
    ta = torch.tensor(scaler.action(actions), dtype=torch.float32).to(device)
    print(f"{len(paths)} demos -> {len(states)} windows "
          f"(state {tuple(ts.shape[1:])}, action {tuple(ta.shape[1:])}) on {device}")
    # the yardstick a noise_std has to be read against: how far the ODE actually travels
    # from each seed, in the normalized units noise_std is quoted in
    x0, x1 = ta[:, :CHUNK], ta[:, CHUNK:]
    print(f"transport |x1 - x0|: a2a {(x1 - x0).norm(dim=-1).mean():.3f}, "
          f"noise {(x1 - torch.randn_like(x1)).norm(dim=-1).mean():.3f}")
    plot_data(paths, labels, f"{args.out}_data.png")

    arms = ARMS if args.only_arm is None else [a for a in ARMS if a[0] == args.only_arm]
    if not arms:
        raise SystemExit(f"--only-arm {args.only_arm!r} not found; choices: "
                          f"{[a[0] for a in ARMS]}")

    print(f"\n{'arm':<10} {'loss':>7} {'left':>6} {'right':>6} {'wide':>6} "
          f"{'success':>8} {'x spread':>9}")
    results = []
    for name, std in arms:
        model, loss_trace = train(args.model, std, ts, ta, args.train_steps, args.batch,
                                   args.lr, 10, block_size=args.block_size,
                                   compile_model=args.compile)
        roll = rollout(model, scaler, args.rollouts, args.max_chunks,
                       np.random.default_rng(args.seed + 1), trace=args.trace,
                       repeat_last=args.repeat_last, exec_noise=args.exec_noise)
        if args.trace:
            out_path = f"{args.out}_{name}_trace.npy"
            np.save(out_path, roll)
            d = np.linalg.norm(np.diff(roll, axis=1), axis=-1)
            print(f"wrote {out_path}  shape {roll.shape}  "
                  f"step |delta|: mean {d.mean():.3f} max {d.max():.3f}")
        modes = classify(roll)
        stats = summarize(name, modes)
        print(f"{name:<10} {loss_trace[-1]:>7.4f} {stats['left']:>6.0%} {stats['right']:>6.0%} "
              f"{stats['wide']:>6.0%} {stats['success']:>8.0%} {stats['spread']:>9.3f}")
        results.append((name, roll, modes, stats))

    plot_rollouts(results, f"{args.out}_rollouts.png")


if __name__ == "__main__":
    main()
