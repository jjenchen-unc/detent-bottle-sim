"""Run one episode of a policy in DetentBottle-v0 at a given bottle pose and summarise it."""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from detent_bottle.gym_env import DetentBottleEnv
from .expert import ScriptedExpert

POSE_DIMS = np.r_[0:6, 7:13]   # arm translation/rotation action dims (not the jaws)
TIPPED_DEG = 60.   # the env only flags a fall below z = .76 m, so a bottle lying on the table runs to the time limit


def tilt_deg(sim):
    return float(np.degrees(np.arccos(np.clip(sim.data.body('bottle').xmat.reshape(3, 3)[2, 2], -1, 1))))


def rollout(policy, pose, *, cameras=(), image_size=128, max_steps=2400, record=False, noise=0., noise_every=0, noise_len=1,
            seed=0, stop_tipped=True):
    """policy(obs, sim) -> 14-D action. With noise > 0, Gaussian noise of that std is added to the
    executed arm motion (not the jaws) while the clean policy action is what gets recorded.
    noise_every=0 adds it every tick; noise_every=N adds it in bursts of noise_len ticks that start
    on average every N ticks, so a settle expert can still see the robot come to rest in between.
    With stop_tipped, the episode ends as 'bottle_tipped' once the bottle leans past TIPPED_DEG
    (it can no longer meet the upright success condition). Returns (summary dict, episode dict or None)."""
    t = time.time()
    rng = np.random.default_rng(seed)
    env = DetentBottleEnv(image_size=image_size, cameras=cameras, max_steps=max_steps, bottle_pose=tuple(pose))
    obs, info = env.reset(seed=seed)
    if hasattr(policy, 'reset'): policy.reset(env.sim)
    steps, max_lid, max_loop, burst = [], 0., 0., 0
    while True:
        action = np.asarray(policy(obs, env.sim), np.float32)
        if record:
            s = env.sim
            steps.append(dict(obs=obs, action=action, bottle_qpos=info['privileged']['bottle_qpos'],
                              target_pos=s.targets.copy(), target_rotvec=Rotation.from_matrix(s.rotations).as_rotvec(),
                              stage=getattr(policy, 'stage', '')))
        executed = action.copy()
        if noise_every and not burst and rng.random() < 1 / noise_every: burst = noise_len
        if noise and (not noise_every or burst):
            executed[POSE_DIMS] += rng.normal(0, noise, len(POSE_DIMS))
            burst = max(burst - 1, 0)
        obs, reward, terminated, truncated, info = env.step(np.clip(executed, -1, 1))
        max_lid, max_loop = max(max_lid, info['privileged']['lid_deg']), max(max_loop, info['privileged']['loop_deg'])
        reason = info['termination_reason']
        if stop_tipped and not reason and tilt_deg(env.sim) > TIPPED_DEG: reason = 'bottle_tipped'
        if reason: break
    tilt = tilt_deg(env.sim)
    summary = dict(pose=[float(v) for v in pose], success=bool(info['success']), reason=reason,
                   steps=int(info['step']), max_lid_deg=round(max_lid, 1), max_loop_deg=round(max_loop, 1),
                   final_tilt_deg=round(float(tilt), 1),
                   bottle_drift_mm=round(1000 * float(np.linalg.norm(info['privileged']['bottle_pos'][:2] - env.sim.start_bottle[:2])), 1),
                   wall_s=round(time.time() - t, 1))
    episode = None
    if record:
        episode = {f'obs/{k}': np.stack([s['obs'][k] for s in steps]) for k in steps[0]['obs']}
        episode.update({k: np.stack([s[k] for s in steps]) for k in ('action', 'bottle_qpos', 'target_pos', 'target_rotvec')})
        episode['stage'] = np.array([s['stage'] for s in steps])
    env.close()
    return summary, episode


def expert_policy(pose, settle=False):
    return ExpertPolicy(ScriptedExpert(pose, settle))


class ExpertPolicy:
    """Gym-style wrapper around ScriptedExpert (reads the env's Cartesian targets; obs only to detect settling)."""

    def __init__(self, expert):
        self.expert = expert

    def __call__(self, obs, sim):
        return self.expert(sim, obs)

    def reset(self, sim):
        self.expert.reset(sim)

    @property
    def stage(self):
        return self.expert.stage


if __name__ == '__main__':
    import sys
    pose = [float(v) for v in sys.argv[1:4]] if len(sys.argv) > 3 else [0., 0., 0.]
    print(rollout(expert_policy(pose, settle='--settle' in sys.argv), pose)[0])
