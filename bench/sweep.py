"""Run the pose-aware expert over a list of bottle poses in parallel; one JSON line per pose.

python -m bench.sweep --axes x y yaw              # one-axis-at-a-time probe of the reachable range
python -m bench.sweep --random 60 --x .04 --y .04 --yaw 20 --seed 1
"""
import argparse
import json
import os
from multiprocessing import Pool
import numpy as np
from .rollout import expert_policy, rollout


def run(pose):
    return rollout(expert_policy(pose), pose)[0]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--axes', nargs='*', default=[], choices=['x', 'y', 'yaw'])
    p.add_argument('--random', type=int, default=0)
    p.add_argument('--x', type=float, default=.04); p.add_argument('--y', type=float, default=.04); p.add_argument('--yaw', type=float, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 1))
    p.add_argument('--out', default='runs/sweep.jsonl')
    args = p.parse_args()
    poses = []
    for axis in args.axes:
        i = ['x', 'y', 'yaw'].index(axis)
        for v in ([(-.08, -.06, -.04, -.02, .02, .04, .06, .08)] * 2 + [(-45, -30, -20, -10, 10, 20, 30, 45)])[i]:
            pose = [0., 0., 0.]; pose[i] = v; poses.append(pose)
    if args.random:
        rng = np.random.default_rng(args.seed)
        poses += (rng.uniform(-1, 1, (args.random, 3)) * [args.x, args.y, args.yaw]).round(4).tolist()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with Pool(args.workers) as pool, open(args.out, 'a') as f:
        for summary in pool.imap_unordered(run, poses):
            f.write(json.dumps(summary) + '\n'); f.flush()
            print(summary, flush=True)
