"""Conversions between DetentBottle-v0 delta actions and absolute Cartesian tool targets.

The env keeps a persistent tool target per arm and each action nudges it (see README). A policy
that wants absolute targets can track them itself: they start at START_POS / START_ROT on every
reset and change only through apply_action(), which mirrors DetentBottleEnv.step exactly.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from detent_bottle.core import WORKSPACE
from detent_bottle.gym_env import MAX_JAW, POS_SCALE, ROT_SCALE

START_POS = np.array([[-.12, -.18, 1.12], [.13, .16, 1.12]])   # DetentBottleSim.reset targets
START_ROT = np.array([np.diag([1., -1., -1.])] * 2)
ABS_DIM = 20   # per arm: position (3), rotation as its first two columns (6), jaw command (1)


def unit_clip(v):
    """Scale v so its largest component is within [-1, 1], keeping its direction."""
    return v / max(1., np.abs(v).max())


def jaw_command(gap):
    """Jaw action value for a finger gap in metres (0.09 open -> -1, 0 closed -> +1)."""
    return 1 - 2 * gap / MAX_JAW


def delta_action(goal_pos, goal_rot, jaw, pos, rot):
    """14-D action moving the current targets (pos (2, 3), rot (2, 3, 3)) toward the goals as fast
    as the action scale allows; jaw (2,) is passed through."""
    a = np.zeros((2, 7))
    for i in range(2):
        a[i, :3] = unit_clip((np.clip(goal_pos[i], *WORKSPACE) - pos[i]) / POS_SCALE)
        a[i, 3:6] = unit_clip(Rotation.from_matrix(goal_rot[i] @ rot[i].T).as_rotvec() / ROT_SCALE)
        a[i, 6] = jaw[i]
    return a.ravel().astype(np.float32)


def apply_action(action, pos, rot):
    """The env's target update for one action -> (pos, rot)."""
    a = np.clip(np.asarray(action, float).reshape(2, 7), -1, 1)
    pos = np.clip(pos + a[:, :3] * POS_SCALE, *WORKSPACE)
    rot = Rotation.from_rotvec(a[:, 3:6] * ROT_SCALE).as_matrix() @ rot
    return pos, rot


def to_abs(pos, rot, jaw):
    """(..., 2, 3), (..., 2, 3, 3), (..., 2) -> (..., 20)."""
    cols = np.swapaxes(rot[..., :, :2], -1, -2).reshape(*rot.shape[:-2], 6)
    return np.concatenate([pos, cols, jaw[..., None]], -1).reshape(*pos.shape[:-2], ABS_DIM)


def from_abs(x):
    """(20,) -> pos (2, 3), rot (2, 3, 3) re-orthonormalised by Gram-Schmidt, jaw (2,)."""
    x = np.asarray(x, float).reshape(2, 10)
    a, b = x[:, 3:6], x[:, 6:9]
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (a * b).sum(-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return x[:, :3], np.stack([a, b, np.cross(a, b)], -1), np.clip(x[:, 9], -1, 1)
