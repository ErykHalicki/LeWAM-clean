"""Replay lerobot/pusht action trajectories in the swm absolute-action PushT env, to sanity
-check that the recorded absolute-pixel actions drive the swm env the same way (and to eye
the train->eval appearance gap for a pixel policy).

For each episode it matches the initial agent position (lerobot records only agent xy, not
block pose, so the swm block is placed arbitrarily and its interaction will differ from the
demo), replays the recorded actions in swm/PushTFixedGoalAbsoluteAction-v1, and writes per
episode: a montage PNG (lerobot vs swm frames over time + agent-path overlay) and a
side-by-side GIF.

Run:  python -m lewam.scripts.replay_lerobot_pusht --episodes 0,1 --out-dir /tmp/replay
"""

import argparse
from pathlib import Path

import numpy as np


def _to_hwc_uint8(img):
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8 and a.max() <= 1.0:
        a = (a * 255).astype(np.uint8)
    return a.astype(np.uint8)


def _resize(img, size):
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize((size, size), Image.NEAREST))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default="lerobot/pusht")
    p.add_argument("--env", default="swm/PushTFixedGoalAbsoluteAction-v1")
    p.add_argument("--episodes", default="0,1", help="comma-separated episode indices")
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--resolution", type=int, default=224, help="swm render size")
    p.add_argument("--block-init", default="256,300", help="arbitrary swm block xy (lerobot lacks it)")
    p.add_argument("--fps", type=int, default=15, help="gif frame rate")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--no-gif", action="store_true")
    args = p.parse_args()

    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import gymnasium as gym
    import lewam.envs  # noqa: F401 -- registers the swm env variants
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    episodes = [int(e) for e in args.episodes.split(",")]
    bx, by = (float(v) for v in args.block_init.split(","))

    meta = LeRobotDatasetMetadata(args.repo_id)
    ds = LeRobotDataset(args.repo_id)
    env = gym.make(args.env, resolution=args.resolution)

    for ep in episodes:
        lo = int(meta.episodes[ep]["dataset_from_index"])
        hi = min(int(meta.episodes[ep]["dataset_to_index"]), lo + args.max_steps)
        idxs = list(range(lo, hi))
        lr_imgs = [_to_hwc_uint8(ds[i]["observation.image"]) for i in idxs]
        actions = np.stack([np.asarray(ds[i]["action"], dtype=np.float32) for i in idxs])
        lr_agent = np.stack([np.asarray(ds[i]["observation.state"], dtype=np.float32) for i in idxs])

        a0 = lr_agent[0]
        env.reset(seed=ep, options={"state": np.array([a0[0], a0[1], bx, by, 0.0, 0.0, 0.0])})
        swm_imgs, swm_agent = [], []
        for a in actions:
            env.step(a.astype(np.float32))
            swm_imgs.append(np.asarray(env.render(), dtype=np.uint8))
            swm_agent.append(np.asarray(env.unwrapped.agent.position, dtype=np.float32))
        swm_agent = np.stack(swm_agent)

        cols = np.linspace(0, len(idxs) - 1, 7).astype(int)
        fig = plt.figure(figsize=(14, 6.2))
        gs = fig.add_gridspec(3, 7, height_ratios=[1, 1, 1.3], hspace=0.25, wspace=0.05)
        for c, t in enumerate(cols):
            ax = fig.add_subplot(gs[0, c]); ax.imshow(lr_imgs[t]); ax.axis("off"); ax.set_title(f"t={t}", fontsize=8)
            if c == 0: ax.set_ylabel("lerobot", fontsize=9)
            ax2 = fig.add_subplot(gs[1, c]); ax2.imshow(swm_imgs[t]); ax2.axis("off")
            if c == 0: ax2.set_ylabel("swm replay", fontsize=9)
        axp = fig.add_subplot(gs[2, :])
        axp.plot(lr_agent[:, 0], lr_agent[:, 1], "-o", ms=2, label="lerobot agent", color="tab:blue")
        axp.plot(swm_agent[:, 0], swm_agent[:, 1], "-o", ms=2, label="swm agent (replay)", color="tab:red")
        axp.set_xlim(0, 512); axp.set_ylim(512, 0); axp.set_aspect("equal")
        axp.legend(fontsize=8); axp.set_title("agent path: lerobot vs swm-replay (pixel coords)", fontsize=9)
        fig.suptitle(f"{args.repo_id} episode {ep} replayed in {args.env}", fontsize=10)
        png = out / f"replay_ep{ep}.png"
        fig.savefig(png, bbox_inches="tight", dpi=90); plt.close(fig)

        if not args.no_gif:
            r = args.resolution
            frames = []
            for lr, sw in zip(lr_imgs, swm_imgs):
                pair = np.concatenate([_resize(lr, r), np.full((r, 4, 3), 255, np.uint8), _resize(sw, r)], axis=1)
                frames.append(pair)
            gif = out / f"replay_ep{ep}.gif"
            imageio.mimsave(gif, frames, duration=1.0 / args.fps, loop=0)
            print(f"wrote {png} and {gif}  (left: lerobot, right: swm replay)")
        else:
            print(f"wrote {png}")

    env.close()


if __name__ == "__main__":
    main()
