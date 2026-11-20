"""In-memory demonstration dataset for chunked absolute-action policies.

Labels are the absolute tool targets the expert asked for at each tick (bench.actions.to_abs of
the target after the clean action), so a chunk never accumulates delta-integration error.
Images are kept only every `stride` ticks to save memory; labels stay at full rate.
"""
import glob
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation
from ..actions import apply_action, to_abs

PROPRIO = ('joint_pos', 'gripper_qpos', 'finger_force')


def abs_labels(ep):
    rot = Rotation.from_rotvec(ep['target_rotvec'].reshape(-1, 3)).as_matrix().reshape(-1, 2, 3, 3)
    out = []
    for a, p, r in zip(ep['action'], ep['target_pos'], rot):
        p2, r2 = apply_action(a, p, r)
        out.append(to_abs(p2, r2, np.array([a[6], a[13]])))
    return np.array(out, np.float32)


class Demos:
    def __init__(self, roots, cameras, chunk, stride=2, proprio=PROPRIO, episodes=None):
        files = episode_files(roots)
        if episodes is not None: files = [files[i] for i in episodes if i < len(files)]
        if not files: raise FileNotFoundError(f'no episode_*.npz in {roots}')
        self.cameras, self.chunk, self.proprio_keys = tuple(cameras), chunk, tuple(proprio)
        images, proprios, labels, masks, index = [], [], [], [], []
        for e, f in enumerate(files):
            ep = dict(np.load(f))
            T = len(ep['action'])
            keep = np.arange(0, T, stride)
            images.append(np.stack([ep['obs/' + c][keep] for c in self.cameras], 1))   # (n, C, H, W, 3)
            proprios.append(np.concatenate([ep['obs/' + k][keep].reshape(len(keep), -1) for k in self.proprio_keys], 1))
            lab = abs_labels(ep)
            labels.append(np.concatenate([lab, np.repeat(lab[-1:], chunk, 0)]).astype(np.float32))
            masks.append(np.r_[np.ones(T, bool), np.zeros(chunk, bool)])
            index += [(e, i, t) for i, t in enumerate(keep)]
        self.images, self.proprios, self.labels, self.masks = images, proprios, labels, masks
        self.index = np.array(index)
        self.files = [os.path.relpath(f) for f in files]
        allp, alll = np.concatenate(proprios), np.concatenate([l[:-chunk] for l in labels])
        self.stats = dict(proprio_mean=allp.mean(0), proprio_std=np.maximum(allp.std(0), 1e-2),
                          action_mean=alll.mean(0), action_std=np.maximum(alll.std(0), 1e-2))
        self.image_size = images[0].shape[2]
        self.proprio_dim, self.action_dim = allp.shape[1], alll.shape[1]

    def __len__(self):
        return len(self.index)

    def batch(self, rows, rng):
        """rows: array of index rows -> dict of numpy arrays (images uint8, normalised proprio/actions)."""
        s = self.stats
        img = np.stack([self.images[e][i] for e, i, _ in rows])
        prop = np.stack([self.proprios[e][i] for e, i, _ in rows])
        act = np.stack([self.labels[e][t:t + self.chunk] for e, _, t in rows])
        mask = np.stack([self.masks[e][t:t + self.chunk] for e, _, t in rows])
        return dict(images=img, proprio=(prop - s['proprio_mean']) / s['proprio_std'],
                    actions=(act - s['action_mean']) / s['action_std'], mask=mask)

    def sample(self, n, rng):
        return self.batch(self.index[rng.integers(0, len(self.index), n)], rng)


def episode_files(roots):
    """Sorted episode files of one demo directory or a list of them."""
    roots = [roots] if isinstance(roots, str) else roots
    return [f for r in roots for f in sorted(glob.glob(os.path.join(r, 'episode_*.npz')))]


def split_episodes(roots, val=2, seed=0):
    n = len(episode_files(roots))
    order = np.random.default_rng(seed).permutation(n)
    return sorted(order[val:].tolist()), sorted(order[:val].tolist())


def load_config(root):
    with open(os.path.join(root, 'config.json')) as f: return json.load(f)
