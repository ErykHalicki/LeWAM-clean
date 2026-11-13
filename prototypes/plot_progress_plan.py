"""Plots for proto_progress_plan.py output.

  python prototypes/plot_progress_plan.py prototypes/out/progress_plan.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    src = Path(sys.argv[1])
    d = json.loads(src.read_text())
    S = d["states"]
    ks = d["ks"]
    lams = d["lams"]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    for k in ks:
        xs, ys, es = [], [], []
        for lam in lams:
            rows = [r for r in S if r["k"] == k and r["lam"] == lam]
            dm = np.array([r["dmse"] for r in rows])
            xs.append(lam)
            ys.append(dm.mean())
            es.append(dm.std() / np.sqrt(len(dm)))
        ax[0].errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                       label=f"k={k} (+{12 * (k + 1)}f)")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("trust-region weight lambda")
    ax[0].set_ylabel("mean dmse (action MSE final - start)")
    ax[0].set_title(f"{d['ckpt']} step {d['step']}: dmse vs lambda\n"
                    f"(<0 = surrogate moved chunk toward expert)")
    ax[0].legend()

    cmap = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
    for k in ks:
        rows = [r for r in S if r["k"] == k and r["lam"] == lams[0]]
        ax[1].scatter([r["surr_final"] for r in rows],
                      [r["mse_final"] for r in rows],
                      c=cmap.get(k, None), s=18, alpha=0.7,
                      label=f"k={k}")
    ax[1].set_xlabel("surrogate (progress) final")
    ax[1].set_ylabel("action MSE to expert final")
    ax[1].set_title(f"surr_final vs mse_final  (lambda={lams[0]})\n"
                    "up-right = Goodhart (high surrogate, high error)")
    ax[1].legend()

    fig.tight_layout()
    out = src.with_suffix(".png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
