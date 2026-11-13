# debug

Diagnostics and one-off debugging utilities. Not part of the training/eval path.

- `probe_dynamics.py` — does a world model's predicted future track the future?
  Scores every predicted non-action stream of a checkpoint on held-out windows against
  model-free baselines (persistence, mean latent, random other window, random vector)
  and against its own prediction under swapped actions (`action_sens`: 0 = the head
  ignores the actions). `python -m lewam.debug.probe_dynamics --run <id>[:best] ...`;
  takes wandb runs or local `.pt` files, runs on CPU.
