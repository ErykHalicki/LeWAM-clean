"""Can SIGReg replace the (deleted) gradient detaches on the action codec?

Two failure modes are in play for the chunk codec (act_enc/act_dec), and the question is
which guard -- `ae_recon` (A2A's L_AE) or a `sigreg` term on the action latent -- actually
prevents each:

  1. SCALE collapse: the codec latent shrinking far below the seed noise injected into it,
     which ae_recon does not prevent. L1 reconstruction constrains INFORMATION, not scale:
     act_dec can invert an arbitrarily small latent by carrying arbitrarily large weights.
     SIGReg fits the latent to N(0, I) -- a statement ABOUT scale -- so it should be the
     guard that pins it.

  2. CONDITIONAL collapse: act_enc(prev chunk) drifting onto act_enc(future chunk) for the
     same episode, leaving the flow nothing to transport. SIGReg flattens the token bag to
     (B*N, D) and never sees which token came from which episode, so an encoder can be
     conditionally collapsed and perfectly isotropic at once (predictive_model.py says as
     much in its _variance_logs docstring). ae_recon, by contrast, must tell chunks apart
     to reconstruct them.

Each experiment trains the real chunk_mlp codec under an explicit adversary loss standing
in for the degenerate incentive, with each guard (at its config-realistic weight) trying to
hold the line, and prints the trajectory so you can see not just whether a guard holds but
how fast the unguarded runs fail.

Run on any box (auto-selects cuda); ~2-4 min on one GPU. All prints flush immediately.
PROBE_STEPS=<n> overrides the 400 adversary steps for a quick smoke run.
"""

import os
import time

import torch
import torch.nn.functional as F

from lewam.algorithms.encoders import build_encoder
from lewam.algorithms.losses import SIGReg

CHUNK, ADIM, LATENT, B = 36, 7, 512, 256
STEPS = int(os.environ.get("PROBE_STEPS", 400))
LOG_EVERY = max(1, STEPS // 4)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu":
    # these kernels are tiny; the default one-thread-per-core explodes into scheduler
    # contention on a many-core box
    torch.set_num_threads(8)

# the shipped codec, verbatim from configs/algorithm/wam_a2a_b601.yaml
ENC = {"type": "chunk_mlp", "in_dim": ADIM, "in_steps": CHUNK, "out_dim": LATENT,
       "out_steps": 1, "hidden_dim": 1024, "depth": 2, "norm": "layer"}
DEC = {"type": "chunk_mlp", "in_dim": LATENT, "in_steps": 1, "out_dim": ADIM,
       "out_steps": CHUNK, "hidden_dim": 1024, "depth": 2, "norm": "layer"}


def say(*a):
    print(*a, flush=True)


def fresh():
    """Same init for every arm; data comes from CPU generators so it is device-invariant."""
    torch.manual_seed(0)
    enc, dec = build_encoder(ENC).to(DEV), build_encoder(DEC).to(DEV)
    sigreg = SIGReg().to(DEV)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=1e-3)
    return enc, dec, sigreg, opt


def chunks(n, gen):
    """Smooth 7-DoF random-walk chunks with a per-episode offset (std ~1, like norm'd data)."""
    walk = torch.randn(n, CHUNK, ADIM, generator=gen).cumsum(dim=1) * 0.1
    return walk + torch.randn(n, 1, ADIM, generator=gen)


def episodes(n, gen):
    """2*CHUNK-step trajectories -> (prev chunk, future chunk): correlated but genuinely
    different, as consecutive real chunks are."""
    walk = torch.randn(n, 2 * CHUNK, ADIM, generator=gen).cumsum(dim=1) * 0.1
    walk = walk + torch.randn(n, 1, ADIM, generator=gen)
    return walk[:, :CHUNK], walk[:, CHUNK:]


def label(guards):
    return " + ".join(f"{g} w={w:g}" for g, w in guards.items()) or "none"


# Experiment 1: SCALE collapse. Adversary: minimize ||z||^2 (shrink the latent).

def scale_metrics(enc, dec, sigreg, xe):
    with torch.no_grad():
        z = enc(xe)
        return z.std().item(), F.l1_loss(dec(z), xe).item(), sigreg(z).item()


def run_scale(guards):
    gen = torch.Generator().manual_seed(0)
    enc, dec, sigreg, opt = fresh()
    xe = chunks(B, torch.Generator().manual_seed(1)).to(DEV)      # fixed eval batch
    t0 = time.time()
    for step in range(STEPS + 1):
        if step % LOG_EVERY == 0:
            std, recon, s = scale_metrics(enc, dec, sigreg, xe)
            say(f"    step {step:>4}: latent std {std:>9.3e}  recon L1 {recon:.4f}  sigreg {s:>9.2f}")
        if step == STEPS:
            break
        x = chunks(B, gen).to(DEV)
        z = enc(x)
        loss = z.square().mean()                                  # the adversary
        if "ae_recon" in guards:
            loss = loss + guards["ae_recon"] * F.l1_loss(dec(z), x)
        if "sigreg" in guards:
            loss = loss + guards["sigreg"] * sigreg(z)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return (*scale_metrics(enc, dec, sigreg, xe), time.time() - t0)


# Experiment 2: CONDITIONAL collapse. Adversary: minimize ||z_prev - z_fut||^2.

def cond_metrics(enc, dec, sigreg, pe, fe):
    with torch.no_grad():
        z_prev, z_fut = enc(pe), enc(fe)
        within = (z_prev - z_fut).norm(dim=-1).mean().item()      # same-episode distance
        across = torch.cdist(z_prev.squeeze(1), z_prev.squeeze(1)).mean().item()
        s = sigreg(torch.cat([z_prev, z_fut], dim=1)).item()
        recon = F.l1_loss(dec(z_prev), pe).item()
        return within, across, z_prev.std().item(), s, recon


def run_cond(guards):
    gen = torch.Generator().manual_seed(0)
    enc, dec, sigreg, opt = fresh()
    egen = torch.Generator().manual_seed(1)
    pe, fe = (t.to(DEV) for t in episodes(B, egen))               # fixed eval batch
    t0 = time.time()
    for step in range(STEPS + 1):
        if step % LOG_EVERY == 0:
            within, across, std, s, recon = cond_metrics(enc, dec, sigreg, pe, fe)
            sep = within / across if across > 0 else float("nan")
            say(f"    step {step:>4}: separation {sep:.4f}  (within {within:>9.3e} / across {across:>9.3e})"
                f"  latent std {std:>9.3e}  sigreg {s:>9.2f}  recon L1 {recon:.4f}")
        if step == STEPS:
            break
        prev, fut = (t.to(DEV) for t in episodes(B, gen))
        z_prev, z_fut = enc(prev), enc(fut)
        loss = (z_prev - z_fut).square().mean()                   # the adversary
        if "ae_recon" in guards:
            loss = loss + guards["ae_recon"] * (F.l1_loss(dec(z_prev), prev)
                                                + F.l1_loss(dec(z_fut), fut))
        if "sigreg" in guards:
            loss = loss + guards["sigreg"] * sigreg(torch.cat([z_prev, z_fut], dim=1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return (*cond_metrics(enc, dec, sigreg, pe, fe), time.time() - t0)



say(f"torch {torch.__version__}  device {DEV}"
    + (f" ({torch.cuda.get_device_name(0)})" if DEV == "cuda" else ""))
say(f"codec: {ADIM}x{CHUNK} <-> {LATENT}x1 chunk_mlp | B={B} | {STEPS} adversary steps\n")

say("=" * 90)
say("EXPERIMENT 1 -- scale collapse: adversary shrinks ||z||^2; who pins the scale?")
say("=" * 90)
rows = []
for guards in ({}, {"ae_recon": 1.0}, {"sigreg": 0.1}, {"sigreg": 1.0},
               {"ae_recon": 1.0, "sigreg": 0.1}):
    say(f"\n  [{label(guards)}]")
    rows.append((label(guards), *run_scale(guards)))
say(f"\n{'guard':<28}{'latent std':>12}{'recon L1':>10}{'sigreg':>10}{'sec':>7}")
say("-" * 67)
for name, std, recon, s, dt in rows:
    say(f"{name:<28}{std:>12.3e}{recon:>10.4f}{s:>10.2f}{dt:>7.1f}")
say(f"\naction data std for reference: {chunks(B, torch.Generator().manual_seed(1)).std():.3f}")
say("read: latent std ~1e-5 is the collapse; recon L1 can stay LOW through it (scale-blind).")
say("sigreg null is ~1.05; huge values = tiny-scale latent, far from N(0,I).")

say("\n" + "=" * 90)
say("EXPERIMENT 2 -- conditional collapse: adversary pulls z_prev onto z_fut; who keeps them apart?")
say("=" * 90)
rows = []
for guards in ({}, {"sigreg": 1.0}, {"sigreg": 10.0}, {"sigreg": 100.0},
               {"ae_recon": 1.0}, {"ae_recon": 1.0, "sigreg": 0.1}):
    say(f"\n  [{label(guards)}]")
    rows.append((label(guards), *run_cond(guards)))
say(f"\n{'guard':<28}{'separation':>11}{'within':>11}{'across':>11}{'latent std':>11}{'sigreg':>9}{'recon L1':>10}{'sec':>7}")
say("-" * 98)
for name, within, across, std, s, recon, dt in rows:
    sep = within / across if across > 0 else float("nan")
    say(f"{name:<28}{sep:>11.4f}{within:>11.3e}{across:>11.3e}{std:>11.3e}{s:>9.2f}{recon:>10.4f}{dt:>7.1f}")
say("\nread: separation = within-episode / across-episode distance; -> 0 is the collapse the")
say("flow cannot recover from (nothing to transport), ~1 is healthy. If sigreg arms sit near")
say("1.05 on sigreg while separation -> 0, isotropy and conditional collapse coexist -- sigreg")
say("cannot be the guard for THIS failure. Watch ae_recon arms: reconstruction must tell the")
say("two chunks apart, so it should hold separation up.")
