"""V-JEPA 2-AC: the action-only predictor, the teacher-forced + rollout loss, latent
rollouts, CEM planning, and the robot driver, all on a random vit_tiny so no weights are
downloaded."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from lewam.algorithms import build_algorithm
from lewam.experiments.eval.chunking import PlanDriver, make_driver
from lewam.planning.cem import cem

FIELD = "observation.images.zed_left"
pytest.importorskip("timm")


def ac_cfg(**over):
    cfg = {
        "name": "vjepa2_ac",
        "predictor": {"dim": 64, "depth": 2, "heads": 4},
        "loss": {"auto_steps": 2, "normalize_reps": True},
        "encoders": {"vjepa": {"type": "vjepa2", "arch": "vit_tiny", "checkpoint": None,
                               "crop_size": 32, "frames_per_step": 1, "per_frame": True,
                               "frozen": True}},
        "conditioning": {
            "scene_context": {"from": FIELD, "encoder": "vjepa", "via": "tokens", "dim": 192,
                              "index": "-1..0", "raw_index": "-12, 0", "fps": 2.5,
                              "grid": [2, 2]},
            "action": {"from": "action", "role": "action", "dim": 7, "index": "-12..35",
                       "fps": 30},
        },
        "predict": {
            "scene_future": {"from": FIELD, "encoder": "vjepa", "type": "direct",
                             "input": "scene_context", "dim": 192, "index": "1..3",
                             "raw_index": "12, 24, 36", "fps": 2.5, "grid": [2, 2]},
        },
    }
    cfg.update(over)
    return OmegaConf.create(cfg)


def make_batch(b=3):
    torch.manual_seed(0)
    return {
        FIELD: torch.randint(0, 255, (b, 5, 3, 32, 32), dtype=torch.uint8),
        "action": torch.randn(b, 48, 7),
    }


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_algorithm(ac_cfg())


def test_geometry_is_read_off_the_spec(model):
    assert model.action_stream is None and model.action_field == "action"
    assert (model.frames_ctx, model.frames_future, model.sub) == (2, 3, 12)
    assert model.action_embed_dim == 84 and model.chunk_len == 36 and model.past_steps == 1
    assert model.tokens_per_frame == 4 and model.obs_len == 13
    assert model.field_offsets["action"] == list(range(-12, 36))
    # nothing but the predictor trains
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("predictor.") for n in trainable)
    assert not any(n.startswith("predictor.state_encoder") for n, _ in model.named_parameters())
    assert model.summary()["params/predictor"] == sum(
        p.numel() for p in model.predictor.parameters())


def test_loss_is_tf_plus_ar_and_only_the_predictor_gets_gradients(model):
    model.train()
    out = model.loss(make_batch())
    assert set(out) >= {"loss", "loss/tf", "loss/ar", "var/scene/temporal", "var/scene/spatial"}
    assert torch.isclose(out["loss"], out["loss/tf"] + out["loss/ar"])
    model.zero_grad()
    out["loss"].backward()
    assert all(p.grad is not None for p in model.predictor.parameters())
    assert all(p.grad is None for p in model.encoders.parameters())


def test_rollout_consumes_one_action_per_step_and_is_causal(model):
    model.eval()
    batch = make_batch()
    z = model.encode_frames(batch[FIELD][:, :2])
    assert z.shape == (3, 2, 4, 192)
    past = torch.randn(3, 1, 84)
    cand = torch.randn(3, 3, 84)
    final = model.rollout_latent(z, past, cand)
    every = model.rollout_latent(z, past, cand, keep_all=True)
    assert final.shape == (3, 4, 192) and every.shape == (3, 12, 192)
    assert torch.allclose(every[:, -4:], final)
    # the first predicted frame cannot depend on later candidates
    cand2 = cand.clone()
    cand2[:, 1:] += 1.0
    assert torch.allclose(model.rollout_latent(z, past, cand2, keep_all=True)[:, :4],
                          every[:, :4], atol=1e-5)
    with pytest.raises(ValueError):
        model.rollout_latent(z, torch.randn(3, 2, 84), cand)


def test_predict_returns_teacher_forced_future_latents(model):
    batch = make_batch()
    out = model.predict({FIELD: batch[FIELD][:, :2], "action": batch["action"]})
    assert out.shape == (3, 12, 192)


def test_cem_moves_toward_the_cheap_region():
    target = torch.full((2, 3, 5), 0.5)
    plan = cem(lambda c: ((c - target.repeat_interleave(c.shape[0] // 2, 0)) ** 2).mean((1, 2)),
               torch.zeros(2, 3, 5), {"num_samples": 64, "n_steps": 8, "topk": 8, "clip": 1.0, "init_std": 1.0},
               generator=torch.Generator().manual_seed(0))
    assert plan.shape == (2, 3, 5)
    assert (plan - target).abs().mean() < 0.25
    assert plan.abs().max() <= 1.0
    with pytest.raises(ValueError):
        cem(lambda c: c.sum((1, 2)), torch.zeros(1, 1, 1), {"num_samples": 2, "topk": 4})


def test_plan_returns_a_bounded_chunk(model):
    batch = make_batch()
    z = model.encode_frames(batch[FIELD][:, 2:3])[:, 0]
    plan = model.plan(batch[FIELD][:, :2], z, batch["action"][:, :12],
                      {"space": "chunk", "num_samples": 6, "n_steps": 2, "topk": 2, "clip": 2.0, "chunk": 5})
    assert plan.shape == (3, 36, 7) and plan.abs().max() <= 2.0
    for space in ("hold", "linear"):
        plan = model.plan(batch[FIELD][:, :2], z, batch["action"][:, :12],
                          {"space": space, "num_samples": 6, "n_steps": 2, "topk": 2, "clip": 2.0})
        assert plan.shape == (3, 36, 7)
        # smooth spaces: the 12 rows of a step move monotonically between knots
        assert (plan[:, 1:] - plan[:, :-1]).abs().mean() < 0.1
    # linear plans start where the executed actions left off
    assert torch.allclose(plan[:, 0], batch["action"][:, 11], atol=0.2)
    plan0 = model.plan(batch[FIELD][:, :2], z, None, {"num_samples": 4, "n_steps": 1, "topk": 1})
    assert plan0.shape == (3, 36, 7)


def test_spec_mistakes_are_build_errors():
    with pytest.raises(ValueError, match="frozen"):
        cfg = ac_cfg()
        cfg.encoders.vjepa.frozen = False
        build_algorithm(cfg)
    with pytest.raises(ValueError, match="transitions"):
        cfg = ac_cfg()
        cfg.conditioning.action.index = "-12..34"
        build_algorithm(cfg)
    with pytest.raises(ValueError, match="direct"):
        cfg = ac_cfg()
        cfg.predict.scene_future.pop("type")
        build_algorithm(cfg)


def test_plan_driver_executes_a_planned_chunk(model):
    model.eval()
    model.norm_stats = {"action": {"mean": [0.0] * 7, "std": [2.0] * 7}}
    model.norm_method = "mean_std"
    feats = {FIELD: {"dtype": "video", "names": ["h", "w", "c"]},
             "action": {"dtype": "float32", "names": [f"j{i}" for i in range(7)]}}
    goal = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
    cfg = OmegaConf.create({"image_size": [32, 32], "execute_len": 18})
    driver = PlanDriver(model, cfg, feats, goal,
                        plan={"num_samples": 4, "n_steps": 1, "topk": 2, "clip": 1.0})
    assert driver.offsets == [-12, 0]
    assert driver.action_offsets == list(range(-12, 0))
    assert driver.goal_latent.shape == (1, 4, 192)
    driver.reset()
    actions = [driver.step({FIELD: np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)})
               for _ in range(19)]
    assert np.asarray(actions).shape == (19, 7)
    assert len(driver._buffer) == 17                  # 2 replans of 18, 19 popped
    assert np.abs(np.asarray(actions)).max() <= 2.0   # clip 1.0, unnormalized by std 2
    with pytest.raises(ValueError, match="goal_image"):
        make_driver(model, OmegaConf.create({}), feats)
    model.norm_stats = None
