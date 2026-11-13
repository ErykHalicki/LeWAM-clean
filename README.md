# LeWorldActionModel (LeWAM)

Physical robot experiments from LeWAM. Main project page at https://le-wam.github.io/

Paper: [LeWAM: End-to-End World and Action Modeling with JEPAs](https://minghaofu.com/files/lewam.pdf)
(Fu, Siebert, Halicki, Balestriero, 2026). This repo holds the physical-robot study
(Section 4.4, Appendix D): training LeWAM and the baselines on teleoperation data from a
seven-DoF reBot B601-DM arm, and running them closed-loop on the robot.

## Tasks

| task | dataset | episodes | instruction |
| --- | --- | --- | --- |
| Pick and Place T | [`ehalicki/b601_pusht_pick_and_place`](https://huggingface.co/datasets/ehalicki/b601_pusht_pick_and_place) | 200 | Pick up the T block and place it onto the outline. |
| Push-T | [`ehalicki/b601_pusht`](https://huggingface.co/datasets/ehalicki/b601_pusht) | 100 | Push the T block onto the outline. |
| Tape Stack | [`ehalicki/b601_tape_stack`](https://huggingface.co/datasets/ehalicki/b601_tape_stack) | 100 | Stack the green roll of tape on the white roll of tape. |

Each episode records a ZED-Mini scene view, a wrist camera, 7-D joint-target actions and a
21-D follower state at 30 Hz.

## Results

Successes out of 30 trials per task (Table 2 of the paper).

| method | PnP-T | Push-T | Tape Stack | average | GFLOPs / chunk |
| --- | --- | --- | --- | --- | --- |
| Diffusion Policy | 19/30 | 16/30 | 0/30 | 38.9% | 270.0 |
| π0.5 | 14/30 | 3/30 | 9/30 | 28.9% | 3603.6 |
| LeWAM (frozen V-JEPA 2.1 encoder) | 15/30 | 12/30 | 5/30 | 35.6% | 270.2 |
| LeWAM | 16/30 | 8/30 | 6/30 | 33.3% | 270.2 |

## Installation

Uses [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> LeWAM
cd LeWAM
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
uv venv
source .venv/bin/activate
uv pip install -e ".[all]"      # or a lighter extra, e.g. ".[b601]"
```

Extras: `b601` (the real arm), `baselines` (Diffusion Policy, π0.5), `serve` (inference
server, no lerobot or hardware deps), `vjepa`, `wandb`, `dev`, `all`.

`external/lerobot` is a fork ([ErykHalicki/lerobot](https://github.com/ErykHalicki/lerobot))
carrying the B601 arm and ZED camera backends, so lerobot from PyPI will not drive the
robot. `external/vjepa2` is not installed; it is appended to `sys.path` at import.

## Training

The paper's arms, shown for PnP-T (swap `_pnpt` for `_pusht`):

```bash
python main.py run=selfwam_mot_mse_b601_pnpt           # LeWAM
python main.py run=selfwam_mot_mse_frozen_b601_pnpt    # LeWAM, frozen V-JEPA 2.1 encoder
python main.py run=diffusion_b601_pnpt                 # Diffusion Policy
python main.py run=pi05_frozen_b601_pnpt               # π0.5, action expert only
```

Every runnable arm is one file in [`configs/run/`](configs/README.md). Any config value can
be overridden on the CLI, e.g. `experiment.max_steps=30000 wandb.mode=disabled`.

## Running on the robot

The robot host sends its observation window to a GPU box, receives a 36-action chunk, and
executes the first 18 at 30 Hz before asking again.

```bash
# GPU box
./src/lewam/scripts/serve.sh --port 8080

# robot host
python main.py run=b601_irl_pnpt load=<wandb_run_id|ckpt.pt> eval.server=<gpu-host>:8080
```

`b601_irl` is the generic rollout; `b601_irl_pnpt` adds the PnP-T one-minute episode cap.
Teleoperation, recording and gravity compensation scripts live in
[`src/lewam/scripts/`](src/lewam/scripts/README.md).

## Layout

```
configs/             Hydra groups: run, algorithm, dataset, eval, experiment, embodiment
src/lewam/
  algorithms/        encoders, MoT predictor, losses, lerobot baseline wrappers
  datasets/          LeRobot and stable-worldmodel loaders
  experiments/       training, validation and on-robot eval
  scripts/           robot teleop/record, inference server, plotting
```

---

Uses the [robot-learning-research-template](https://github.com/ErykHalicki/robot-learning-research-template) structure.
