"""ACT policy for bench.evaluate: python -m bench.evaluate --policy bench.act.policy:make_policy --kwargs '{"ckpt": "runs/act/v1/last.pt"}'

Uses only observations. It tracks the env's Cartesian targets itself (they start at
bench.actions.START_POS/START_ROT and change only through its own actions) and converts the
temporally ensembled absolute-target prediction into a delta action each tick.
"""
import numpy as np
import torch
from ..actions import START_POS, START_ROT, apply_action, delta_action, from_abs
from .model import ACT
from .train import prepare_images


class ACTPolicy:
    def __init__(self, ckpt, device='cpu', ensemble_k=.01, query_every=1, threads=1):
        torch.set_num_threads(threads)
        state = torch.load(ckpt, map_location=device, weights_only=False)
        c = self.config = state['config']
        self.model = ACT(len(c['cameras']), c['proprio_dim'], c['action_dim'], c['chunk'], c['image_size'], d=c['d'],
                         pretrained=False, feature_layer=c.get('feature_layer', 4))
        self.model.load_state_dict(state['model']); self.model.to(device).eval()
        self.stats = {k: np.array(v, np.float32) for k, v in state['stats'].items()}
        self.cameras, self.image_size, self.device = tuple(c['cameras']), c['image_size'], device
        self.k, self.query_every = ensemble_k, query_every

    def reset(self, sim=None):
        self.pos, self.rot, self.t, self.chunks = START_POS.copy(), START_ROT.copy(), 0, []

    @torch.no_grad()
    def predict(self, obs):
        s = self.stats
        images = np.stack([obs[c] for c in self.cameras])[None]
        proprio = np.concatenate([np.ravel(obs[k]) for k in self.config['proprio']])
        proprio = torch.as_tensor((proprio - s['proprio_mean']) / s['proprio_std'], dtype=torch.float32, device=self.device)[None]
        pred = self.model(prepare_images(images, self.device), proprio)[0][0].cpu().numpy()
        return pred * s['action_std'] + s['action_mean']

    def __call__(self, obs, sim=None):
        if self.t % self.query_every == 0:
            self.chunks.append((self.t, self.predict(obs)))
        chunk = self.config['chunk']
        self.chunks = [(t0, c) for t0, c in self.chunks if self.t - t0 < chunk]
        preds = np.stack([c[self.t - t0] for t0, c in self.chunks])          # oldest first
        w = np.exp(-self.k * np.arange(len(preds))); w /= w.sum()
        pos, rot, jaw = from_abs((w[:, None] * preds).sum(0))
        action = delta_action(pos, rot, jaw, self.pos, self.rot)
        self.pos, self.rot = apply_action(action, self.pos, self.rot)
        self.t += 1
        return action


def make_policy(pose, ckpt, **kwargs):
    """Factory for bench.evaluate; `pose` is ignored (learned policies see only observations)."""
    policy = ACTPolicy(ckpt, **kwargs)
    policy.reset()
    return policy
