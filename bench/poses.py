"""Bottle pose distributions and the fixed held-out evaluation splits shared by every policy.

Poses are (x m, y m, yaw deg) offsets from the default bottle placement. Training data is drawn
from TRAIN_RANGE; test_id uses the same range with a disjoint seed; test_ood is the shell between
TRAIN_RANGE and OOD_RANGE (at least one coordinate outside the training box). train_wide covers
the whole OOD_RANGE box. A policy trained on it has seen test_ood's range, so its test_ood score
measures coverage, not out-of-distribution generalisation.
"""
import numpy as np

TRAIN_RANGE = np.array([.04, .04, 10.])   # +- half-widths; expert sweep: x/y fine to 8 cm, yaw fails at <= -20 deg
OOD_RANGE = np.array([.06, .06, 20.])
SPLIT_SEEDS = dict(train=0, test_id=1001, test_ood=2002, train_wide=3003)


def sample(n, half_width, seed, exclude=None):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        p = rng.uniform(-1, 1, 3) * half_width
        if exclude is None or np.any(np.abs(p) > exclude): out.append(p)
    return np.round(out, 4).tolist()


def split(name, n=None):
    if name == 'default': return [[0., 0., 0.]]
    if name == 'train': return sample(n or 200, TRAIN_RANGE, SPLIT_SEEDS['train'])
    if name == 'train_wide': return sample(n or 300, OOD_RANGE, SPLIT_SEEDS['train_wide'])
    if name == 'test_id': return sample(n or 30, TRAIN_RANGE, SPLIT_SEEDS['test_id'])
    if name == 'test_ood': return sample(n or 20, OOD_RANGE, SPLIT_SEEDS['test_ood'], exclude=TRAIN_RANGE)
    raise ValueError(f'unknown split {name!r}')
