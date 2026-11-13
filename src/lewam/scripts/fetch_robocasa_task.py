"""Materialize a local root of lerobot/robocasa_target_human_unified holding ONLY the
episodes of the requested task(s): data parquets, the video files those episodes live in,
and meta. Unlike HuggingFaceVLA/libero, this repo's meta is consistent (episode_index ==
position, file mappings verified), so no repair pass -- just select, download,
hardlink into a root, and verify offline construction.

Run:  python -m lewam.scripts.fetch_robocasa_task <TaskName> [<TaskName> ...]
"""
import os, sys, shutil

os.environ.setdefault("HF_LEROBOT_HOME", "/tmp/hf_lerobot")

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

REPO = "lerobot/robocasa_target_human_unified"
HOME = os.environ["HF_LEROBOT_HOME"]
ROOT = f"{HOME}/{REPO}"
CAMS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
tasks = sys.argv[1:]
assert tasks, "pass at least one task name (see meta/tasks.parquet)"

meta_snap = snapshot_download(REPO, repo_type="dataset", allow_patterns=["meta/**"],
                              cache_dir=f"{HOME}/hub")
ep = pq.read_table(f"{meta_snap}/meta/episodes/chunk-000/file-000.parquet").to_pandas()
known = sorted({t for row in ep["tasks"] for t in row})
missing = [t for t in tasks if t not in known]
assert not missing, f"unknown tasks {missing}; known: {known}"
sel = ep[ep["tasks"].map(lambda row: any(t in row for t in tasks))]
episodes = sorted(int(e) for e in sel["episode_index"])
frames = int(sel["length"].sum())
print(f"{len(episodes)} episodes, {frames} frames for {tasks}", flush=True)

patterns = sorted({
    f"data/chunk-{int(r['data/chunk_index']):03d}/file-{int(r['data/file_index']):03d}.parquet"
    for _, r in sel.iterrows()
})
for cam in CAMS:
    patterns.extend(sorted({
        f"videos/observation.images.{cam}/chunk-{int(r[f'videos/observation.images.{cam}/chunk_index']):03d}"
        f"/file-{int(r[f'videos/observation.images.{cam}/file_index']):03d}.mp4"
        for _, r in sel.iterrows()
    }))
print(f"downloading {len(patterns)} files + meta", flush=True)
snap = snapshot_download(REPO, repo_type="dataset", allow_patterns=patterns + ["meta/**"],
                         cache_dir=f"{HOME}/hub")

if os.path.exists(ROOT):
    shutil.rmtree(ROOT)
os.makedirs(os.path.dirname(ROOT), exist_ok=True)
for rel in patterns + [os.path.relpath(os.path.join(dp, f), snap)
                       for dp, _, fs in os.walk(f"{snap}/meta") for f in fs]:
    src, dst = f"{snap}/{rel}", f"{ROOT}/{rel}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.link(os.path.realpath(src), dst)
print("root written:", ROOT, flush=True)

os.environ["HF_HUB_OFFLINE"] = "1"
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(REPO, episodes=episodes)
assert ds.num_episodes == len(episodes), (ds.num_episodes, len(episodes))
assert ds.num_frames == frames, (ds.num_frames, frames)
sample = ds[0]
img = sample["observation.images.robot0_agentview_left"]
print("frames:", ds.num_frames, "| sample image:", tuple(img.shape),
      "| action:", tuple(sample["action"].shape),
      "| state:", tuple(sample["observation.state"].shape), flush=True)
print("DONE", flush=True)
