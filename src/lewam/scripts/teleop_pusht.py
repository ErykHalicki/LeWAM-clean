"""Teleoperate PushT with the mouse and record a demonstration dataset in the HDF5
schema datasets/swm.py reads directly (ep_len/ep_offset + pixels/action/proprio/state).

Move the mouse to steer the agent. Press SPACE to manually end + save the current episode
(use this if the block looks done but the env's own success check doesn't trigger). Press
'r' to discard the current episode and reset early. Close the window to stop and save
whatever was recorded so far.

Run:  python -m lewam.scripts.teleop_pusht --episodes 20 --name pusht_teleop
"""

import argparse
from collections import defaultdict


def _remove_if_schemaless(path):
    # a run that quits before recording leaves a schema-less .h5 that HDF5Writer's
    # append mode cannot reopen
    if not path.exists():
        return
    import h5py

    with h5py.File(path, "r") as f:
        has_schema = "ep_len" in f
    if not has_schema:
        path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name", default="pusht_teleop",
        help="dataset name; written to $STABLEWM_HOME/datasets/<name>.h5",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=None, help="overrides $STABLEWM_HOME")
    args = parser.parse_args()

    import gymnasium as gym
    import numpy as np
    import pygame
    import stable_worldmodel  # noqa: F401  registers swm/* env ids
    from stable_worldmodel.data.formats.hdf5 import HDF5Writer
    from stable_worldmodel.data.utils import get_cache_dir

    datasets_dir = get_cache_dir(args.cache_dir, sub_folder="datasets")
    out_path = datasets_dir / f"{args.name}.h5"
    _remove_if_schemaless(out_path)

    env = gym.make(
        "swm/PushT-v1",
        render_mode="human",
        resolution=args.image_size,
        max_episode_steps=args.max_episode_steps,
    )
    raw = env.unwrapped
    clock = pygame.time.Clock()

    written = 0
    seed = args.seed
    with HDF5Writer(out_path) as writer:
        while written < args.episodes:
            obs, info = env.reset(seed=seed)
            seed += 1
            pixels = env.render()
            episode = defaultdict(list)
            discard = False
            manual_done = False
            terminated = truncated = False

            while not (terminated or truncated):
                quit_requested = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_requested = True
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        discard = True
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        manual_done = True
                if quit_requested:
                    break

                mouse_x, mouse_y = pygame.mouse.get_pos()
                target = np.array([mouse_x, mouse_y], dtype=np.float32)
                action = (target - raw.agent.position) / raw.action_scale
                action = np.clip(action, -1.0, 1.0).astype(np.float32)

                episode["pixels"].append(pixels)
                episode["action"].append(action)
                episode["proprio"].append(np.asarray(obs["proprio"], dtype=np.float32))
                episode["state"].append(np.asarray(obs["state"], dtype=np.float32))

                obs, reward, terminated, truncated, info = env.step(action)
                pixels = env.render()
                clock.tick(args.fps)

                if discard or manual_done:
                    break

            if quit_requested:
                break

            if not discard and len(episode["pixels"]) > 0:
                writer.write_episode(dict(episode))
                written += 1
                status = "success" if terminated else ("manual" if manual_done else "timeout")
                print(f"episode {written}/{args.episodes} ({status})")
            else:
                print("discarded episode")

    env.close()
    print(f"wrote {written} episodes to {out_path}")


if __name__ == "__main__":
    main()
