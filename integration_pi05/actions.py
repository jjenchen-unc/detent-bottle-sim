import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from detent_bottle.core import WORKSPACE


class ActionAdapter:
    def __init__(self, sim):
        self.sim = sim
        self.scratch = mujoco.MjData(sim.model)
        self.clips = 0
        self.last_diagnostics = None

    def apply(self, command, fraction=1):
        s = self.sim
        a = np.asarray(command, float)
        if a.shape != (8,) or not np.isfinite(a).all():
            raise ValueError('Expected finite DROID command (8,)')
        if not np.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError('fraction must be in (0, 1]')
        self.scratch.qpos[:] = s.data.qpos
        lo, hi = s.model.jnt_range[s.ids[0]].T
        pre_joint_pos = s.data.qpos[s.qids[0]].copy()
        clipped_command = np.clip(a[:7], -1, 1)
        raw_joint_target = pre_joint_pos + .2 * clipped_command * fraction
        joint_target = np.clip(raw_joint_target, lo, hi)
        self.scratch.qpos[s.qids[0]] = joint_target
        mujoco.mj_forward(s.model, self.scratch)
        pos = self.scratch.site_xpos[s.tips[0]].copy()
        bounded = np.clip(pos, *WORKSPACE)
        self.clips += int(np.any(pos != bounded))
        rot = self.scratch.site_xmat[s.tips[0]].reshape(3, 3)
        closure = np.clip(a[7], 0, 1)
        gap = .09 * (1 - closure)
        s.target(0, bounded, Rotation.from_matrix(rot).as_euler('xyz', degrees=True), gap)
        # Snapshot each conversion stage without changing live physics or control.
        self.last_diagnostics = {
            'command': a.copy(), 'fraction': float(fraction),
            'pre_joint_pos': pre_joint_pos, 'clipped_joint_command': clipped_command,
            'raw_joint_target': raw_joint_target, 'joint_target': joint_target,
            'joint_action_clipped': a[:7] != clipped_command,
            'joint_limit_clipped': raw_joint_target != joint_target,
            'fk_pos': pos, 'fk_rotation': rot.copy(),
            'bounded_target': bounded.copy(), 'workspace_clipped': pos != bounded,
            'gripper_raw': float(a[7]), 'gripper_closure': float(closure),
            'gripper_clipped': bool(a[7] != closure), 'gripper_gap': float(gap),
        }
        return bounded


def resample(chunk, ticks):
    """Overlap-weighted 15 Hz commands on 20 Hz control intervals."""
    if isinstance(ticks, bool) or not isinstance(ticks, int) or not 1 <= ticks <= 20:
        raise ValueError('ticks must be an integer in [1,20]')
    a = np.asarray(chunk, float)
    if a.shape != (15, 8) or not np.isfinite(a).all():
        raise ValueError('Expected finite pi05_droid chunk (15,8)')
    for tick in range(ticks):
        start, end = tick / 20, (tick + 1) / 20
        weights = np.array([max(0, min(end, (j + 1) / 15) - max(start, j / 15)) * 15 for j in range(15)])
        cmd = np.zeros(8)
        cmd[:7] = weights @ np.clip(a[:, :7], -1, 1)
        cmd[7] = a[min(int(start * 15), 14), 7]
        yield cmd
