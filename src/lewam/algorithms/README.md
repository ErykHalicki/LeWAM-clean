# algorithms

The method layer: models, loss/step, and inference API. May import external packages
(torch, encoders, backend libs) but never from `datasets/`, `experiments/`, or `eval/`.

In this layer, the `PredictiveModel`
base is the only code that reads `from`/`encoder`; the `GenericViTTrunk` it feeds sees
only geometry keys and tensors.

There is one predictor (`name: generic_vit_predictor`). Each predicted stream declares
`type: flow | direct`, and `_forward_train` derives its seeding and target from that:
`flow` seeds noise and targets `x1 - x0`; `direct` reads its `input` stream's tokens and
targets the next latent. Adding a paradigm is a config change, not a subclass.

`name: vjepa2_ac` (`vjepa2_ac.py`) is the V-JEPA 2-AC baseline: a frozen V-JEPA encoder,
Meta's action-conditioned block-causal predictor from `external/vjepa2` over its patch
tokens (state token dropped, `ac_predictor.py`), the teacher-forced + rollout L1 loss, and
CEM planning (`planning/cem.py`) toward a goal frame; `eval=holdout_cem` scores the
planned chunk, `PlanDriver` executes it on the robot.
