"""Pose-aware scripted expert: the reference opener moved into the bottle's frame and driven
through the Gym action space. Privileged: it is told the bottle pose.

The reference opener is timed: each segment lasts a fixed number of ticks, and in most of them
the robot reaches its target early and then sits still (25-46 ticks) until the clock runs out.
A reactive policy trained on that cannot tell when to move on, because every still frame looks
the same. With settle=True a segment ends once both tools are on target and the robot has been
still for SETTLE_TICKS, so the next move is triggered by something visible. The original tick
count stays as the upper bound (contact segments such as the loop fold run to it)."""
import numpy as np
from scipy.spatial.transform import Rotation
from detent_bottle.policy import opening_sequence
from .actions import delta_action, jaw_command

SETTLE_TICKS, MIN_TICKS = 3, 5
JOINT_TOL, GRIP_TOL, TRACK_TOL = 1e-4, 1e-5, 1.5e-3   # rad / tick, m / tick, m


def posed_sequence(pose=(0., 0., 0.)):
    """opening_sequence() rigidly moved from the default bottle pose (0, 0, 0) to (x, y, yaw_deg).
    Yields (stage, arm, tool_xyz, rotation_matrix, jaw_gap_m, ticks)."""
    x, y, yaw = pose
    Rz = Rotation.from_euler('z', yaw, degrees=True)
    for stage, arm, pos, rpy, gap, ticks in opening_sequence():
        R = Rz * Rotation.from_euler('xyz', rpy, degrees=True)
        yield stage, arm, Rz.apply(pos) + [x, y, 0], R.as_matrix(), gap, ticks


class ScriptedExpert:
    """Call with the env's sim (and, for settle=True, the Gym observation) each tick; returns a
    14-D Gym action. Holds still once the plan ends."""

    def __init__(self, pose=(0., 0., 0.), settle=False):
        self.segments = list(posed_sequence(pose))
        self.settle = settle
        self.goals = None

    def reset(self, sim):
        self.seg, self.k, self.still, self.last, self.label = 0, 0, 0, None, self.segments[0][0]
        self.goals = [[sim.targets[a].copy(), sim.rotations[a].copy(), 2 * sim.grip_targets[a]] for a in range(2)]

    @property
    def stage(self):
        """Stage label of the most recent action."""
        return self.label

    def _update_still(self, sim, obs):
        q = np.r_[obs['joint_pos'], obs['gripper_qpos']]
        on_target = np.abs(obs['eef_pos'].reshape(2, 3) - sim.targets).max() < TRACK_TOL
        if self.last is not None and on_target and np.abs(q[:14] - self.last[:14]).max() < JOINT_TOL \
                and np.abs(q[14:] - self.last[14:]).max() < GRIP_TOL:
            self.still += 1
        else:
            self.still = 0
        self.last = q

    def __call__(self, sim, obs=None):
        if self.goals is None: self.reset(sim)
        if self.settle: self._update_still(sim, obs)
        if self.seg < len(self.segments):
            ticks = self.segments[self.seg][5]
            if self.k >= ticks or (self.settle and self.k >= MIN_TICKS and self.still >= SETTLE_TICKS):
                self.seg, self.k, self.still = self.seg + 1, 0, 0
        if self.seg < len(self.segments):
            self.label, arm, pos, R, gap, _ = self.segments[self.seg]
            self.goals[arm] = [pos, R, gap]
            self.k += 1
        pos, rot, gap = zip(*self.goals)
        return delta_action(pos, rot, [jaw_command(g) for g in gap], sim.targets, sim.rotations)
