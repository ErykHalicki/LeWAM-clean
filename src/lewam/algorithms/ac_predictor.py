"""V-JEPA 2-AC predictor with action tokens only.

Meta's `VisionTransformerPredictorAC` (external/vjepa2) interleaves `[action, state,
patches]` per frame under a block-causal mask. Its blocks and attention take the number
of leading conditioning tokens as an argument, so dropping the end-effector state token is
a matter of rebuilding the mask and the per-frame layout; every block, projection and the
3D-RoPE (spatiotemporal on patches, temporal-only on the action token) is Meta's code.
"""

import torch
import torch.nn as nn

from .encoders import vjepa_module


def _meta():
    return vjepa_module("src.models.ac_predictor"), vjepa_module("src.models.utils.modules")


def build_action_only_predictor(**kwargs):
    ac, _ = _meta()

    class ActionOnlyPredictorAC(ac.VisionTransformerPredictorAC):
        """`forward(x, actions)`: x (B, T*H*W, embed_dim) frame tokens, actions (B, T,
        action_embed_dim), one per frame; returns (B, T*H*W, embed_dim), the output at
        frame t being the prediction for frame t+1.
        """

        COND_TOKENS = 1

        def __init__(self, **kw):
            super().__init__(**kw)
            # never in the optimizer: nothing feeds them
            del self.state_encoder
            del self.extrinsics_encoder
            self.attn_mask = None
            if self.is_frame_causal:
                _, modules = _meta()
                self.attn_mask = modules.build_action_block_causal_attention_mask(
                    self.num_frames // self.tubelet_size, self.grid_height, self.grid_width,
                    add_tokens=self.COND_TOKENS,
                )

        def forward(self, x, actions, states=None, extrinsics=None):
            x = self.predictor_embed(x)
            B, N_ctxt, D = x.size()
            P = self.grid_height * self.grid_width
            T = N_ctxt // P
            if actions.shape[1] != T:
                raise ValueError(
                    f"{T} context frames need {T} action tokens, got {actions.shape[1]}"
                )
            a = self.action_encoder(actions).unsqueeze(2)
            x = torch.cat([a, x.view(B, T, P, D)], dim=2).flatten(1, 2)
            attn_mask = None
            if self.attn_mask is not None:
                if x.size(1) > self.attn_mask.size(0):
                    raise ValueError(
                        f"{T} frames exceed the {self.num_frames // self.tubelet_size} the "
                        f"causal mask was built for"
                    )
                attn_mask = self.attn_mask[: x.size(1), : x.size(1)].to(x.device, non_blocking=True)
            kw = dict(mask=None, attn_mask=attn_mask, T=T, H=self.grid_height,
                      W=self.grid_width, action_tokens=self.COND_TOKENS)
            for blk in self.predictor_blocks:
                if self.use_activation_checkpointing and torch.is_grad_enabled():
                    x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False, **kw)
                else:
                    x = blk(x, **kw)
            x = x.view(B, T, self.COND_TOKENS + P, D)[:, :, self.COND_TOKENS:].flatten(1, 2)
            return self.predictor_proj(self.predictor_norm(x))

    from functools import partial

    return ActionOnlyPredictorAC(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
