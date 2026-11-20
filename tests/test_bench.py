import numpy as np
from scipy.spatial.transform import Rotation
from detent_bottle.gym_env import DetentBottleEnv
from detent_bottle.policy import opening_sequence
from bench.actions import START_POS, START_ROT, apply_action, from_abs, to_abs
from bench.expert import posed_sequence
from bench.poses import OOD_RANGE, TRAIN_RANGE, split
from bench.rollout import expert_policy


def test_posed_sequence_identity_at_default_pose():
    for (s0, a0, p0, rpy, g0, t0), (s1, a1, p1, R, g1, t1) in zip(opening_sequence(), posed_sequence()):
        assert (s0, a0, g0, t0) == (s1, a1, g1, t1)
        assert np.allclose(p0, p1) and np.allclose(Rotation.from_euler('xyz', rpy, degrees=True).as_matrix(), R)


def test_actions_mirror_env_targets():
    env = DetentBottleEnv(cameras=(), max_steps=50)
    obs, _ = env.reset()
    assert np.allclose(env.sim.targets, START_POS) and np.allclose(env.sim.rotations, START_ROT)
    policy = expert_policy([.02, -.01, 10.]); policy.reset(env.sim)
    pos, rot = START_POS.copy(), START_ROT.copy()
    rng = np.random.default_rng(0)
    for t in range(20):
        a = np.clip(policy(obs, env.sim) + rng.normal(0, .3, 14).astype(np.float32), -1, 1)
        obs, *_ = env.step(a)
        pos, rot = apply_action(a, pos, rot)
        p, r, j = from_abs(to_abs(pos, rot, a[[6, 13]]))
        assert np.allclose(p, pos) and np.allclose(r, rot) and np.allclose(j, a[[6, 13]])
    assert np.array_equal(pos, env.sim.targets) and np.array_equal(rot, env.sim.rotations)
    env.close()


def test_splits_fixed_and_disjoint():
    assert split('test_id') == split('test_id')
    assert not {tuple(p) for p in split('train')} & {tuple(p) for p in split('test_id')}
    assert all(np.all(np.abs(p) <= TRAIN_RANGE) for p in split('test_id'))
    assert all(np.any(np.abs(p) > TRAIN_RANGE) for p in split('test_ood'))
    tests = {tuple(p) for p in split('test_id') + split('test_ood')}
    assert not {tuple(p) for p in split('train_wide')} & tests
    assert all(np.all(np.abs(p) <= OOD_RANGE) for p in split('train_wide'))


def test_bottle_pose_yaw_is_degrees():
    from detent_bottle.core import DetentBottleSim
    sim = DetentBottleSim(bottle_pose=(.01, -.01, 20.))
    yaw = Rotation.from_quat(sim.data.body('bottle').xquat, scalar_first=True).as_euler('xyz', degrees=True)[2]
    assert abs(yaw - 20.) < 1. and np.allclose(sim.data.body('bottle').xpos[:2], [.01, -.01], atol=2e-3)
