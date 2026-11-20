"""Collect scripted-expert demonstrations at randomised bottle poses, in the Gym env's own
observation/action format. Only successful episodes are kept.

python -m bench.collect --n 100 --out data/expert_v1
python -m bench.collect --n 100 --noise .1 --out data/expert_noise10
python -m bench.collect --n 100 --settle --out data/expert_v2      # no idle waits (see bench/expert.py)
python -m bench.collect --n 200 --settle --split train_wide --out data/expert_v3_wide
python -m bench.collect --n 100 --settle --split train_wide --start 200 --noise .15 --noise-every 30 --noise-len 6 --out data/expert_v3_wide_noise

Each kept episode is data/<name>/episode_XXXX.npz with keys
  obs/<camera>        (T, S, S, 3) uint8
  obs/<proprio key>   (T, ...)      joint_pos, joint_vel, gripper_qpos, eef_pos, eef_quat, wrist_ft, finger_force
  action              (T, 14)       clean expert action in [-1, 1] (the label, even when noise was executed)
  target_pos          (T, 2, 3)     Cartesian tool target per arm before the action (for absolute-action policies)
  target_rotvec       (T, 2, 3)     tool target rotation per arm (axis-angle, world frame)
  bottle_qpos         (T, 10)       privileged object state (not for policy input)
  stage               (T,)          expert stage label
  pose                (3,)          bottle (x, y, yaw_deg)
plus manifest.jsonl (one line per attempt, kept or not) and config.json.
"""
import argparse
import json
import os
from multiprocessing import Pool
import numpy as np
from detent_bottle.gym_env import CAMERAS
from .poses import OOD_RANGE, TRAIN_RANGE, split
from .rollout import expert_policy, rollout


def attempt(job):
    index, pose, args = job
    summary, episode = rollout(expert_policy(pose, args['settle']), pose, cameras=args['cameras'], image_size=args['image_size'],
                               record=True, noise=args['noise'], noise_every=args['noise_every'], noise_len=args['noise_len'], seed=index)
    summary['index'] = index
    if summary['success']:
        path = os.path.join(args['out'], f'episode_{index:04d}.npz')
        np.savez_compressed(path, pose=np.array(pose), **episode)
        summary['file'] = os.path.basename(path); summary['bytes'] = os.path.getsize(path)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=100, help='number of training poses to attempt (first n of the split)')
    p.add_argument('--split', default='train', choices=['train', 'train_wide'], help='pose split to draw from (bench/poses.py)')
    p.add_argument('--start', type=int, default=0, help='skip the first poses of the split (to extend a dataset)')
    p.add_argument('--cameras', nargs='+', default=list(CAMERAS))
    p.add_argument('--image-size', type=int, default=96)
    p.add_argument('--noise', type=float, default=0., help='std of Gaussian noise on executed arm actions')
    p.add_argument('--noise-every', type=int, default=0, help='0: noise every tick; N: bursts starting on average every N ticks')
    p.add_argument('--noise-len', type=int, default=1, help='ticks per noise burst')
    p.add_argument('--settle', action='store_true', help='expert moves on once the robot is still instead of waiting out the clock')
    p.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 1))
    p.add_argument('--out', required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    config = dict(cameras=args.cameras, image_size=args.image_size, noise=args.noise, noise_every=args.noise_every, noise_len=args.noise_len,
                  settle=args.settle, split=args.split, out=args.out)
    with open(os.path.join(args.out, 'config.json'), 'w') as f:
        pose_range = OOD_RANGE if args.split == 'train_wide' else TRAIN_RANGE
        json.dump(dict(config, pose_range=pose_range.tolist(), start=args.start, n=args.n), f, indent=2)
    poses = split(args.split, args.start + args.n)[args.start:]
    kept = 0
    with Pool(args.workers) as pool, open(os.path.join(args.out, 'manifest.jsonl'), 'a') as f:
        for summary in pool.imap_unordered(attempt, [(args.start + i, pose, config) for i, pose in enumerate(poses)]):
            kept += summary['success']
            f.write(json.dumps(summary) + '\n'); f.flush()
            print(f"[{kept} kept] {summary}", flush=True)
    print(f'kept {kept}/{len(poses)} episodes in {args.out}')
