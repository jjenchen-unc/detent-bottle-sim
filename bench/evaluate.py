"""Shared evaluation harness: run any policy on the fixed held-out bottle poses and report
success rate, stage progress and failure reasons.

A policy is given as 'module:function'. The function is called once per episode as
factory(pose, **kwargs) and returns a callable policy(obs, sim) -> 14-D action in [-1, 1].
Learned policies must ignore `pose` and `sim` (they are there for privileged baselines such as
the scripted expert). Optional attributes on the returned policy: `cameras` (tuple of camera
names to render, default none), `image_size` (default 128), `reset(sim)`.

python -m bench.evaluate --policy bench.rollout:expert_policy --split test_id
python -m bench.evaluate --policy my_pkg.dp:make_policy --kwargs '{"ckpt": "ckpt.pt"}' --split test_id test_ood
"""
import argparse
import importlib
import json
import os
import time
from collections import Counter
from multiprocessing import Pool
from .poses import split
from .rollout import rollout

LOOP_OPEN_DEG = 150.   # the detent's open threshold: past this the loop stays down
LID_OPEN_DEG = 60.


def load(spec):
    module, name = spec.split(':')
    return getattr(importlib.import_module(module), name)


def episode(job):
    spec, kwargs, pose, max_steps = job
    policy = load(spec)(pose, **kwargs)
    summary = rollout(policy, pose, cameras=getattr(policy, 'cameras', ()), image_size=getattr(policy, 'image_size', 128),
                      max_steps=max_steps)[0]
    return summary


def report(rows):
    n = len(rows)
    ok = [r for r in rows if r['success']]
    return dict(episodes=n, success_rate=round(len(ok) / n, 3),
                loop_folded_rate=round(sum(r['max_loop_deg'] > LOOP_OPEN_DEG for r in rows) / n, 3),
                lid_opened_rate=round(sum(r['max_lid_deg'] > LID_OPEN_DEG for r in rows) / n, 3),
                mean_success_steps=round(sum(r['steps'] for r in ok) / len(ok), 1) if ok else None,
                reasons=dict(Counter(r['reason'] for r in rows)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--policy', required=True, help="'module:function' policy factory")
    p.add_argument('--kwargs', default='{}', help='JSON kwargs for the factory')
    p.add_argument('--split', nargs='+', default=['test_id'], help='default, test_id, test_ood')
    p.add_argument('--n', type=int, help='use only the first n poses of each split')
    p.add_argument('--max-steps', type=int, default=2400)
    p.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 1))
    p.add_argument('--name', help='results directory name under runs/eval/')
    args = p.parse_args()
    kwargs = json.loads(args.kwargs)
    out = os.path.join('runs', 'eval', args.name or time.strftime('%Y%m%d-%H%M%S-') + args.policy.split(':')[0].split('.')[-1])
    os.makedirs(out, exist_ok=True)
    results = {}
    with Pool(args.workers) as pool:
        for name in args.split:
            poses = split(name)[:args.n]
            rows = []
            with open(os.path.join(out, name + '.jsonl'), 'w') as f:
                for row in pool.imap(episode, [(args.policy, kwargs, pose, args.max_steps) for pose in poses]):
                    rows.append(row); f.write(json.dumps(row) + '\n'); f.flush()
                    print(name, row, flush=True)
            results[name] = report(rows)
    meta = dict(policy=args.policy, kwargs=kwargs, max_steps=args.max_steps, results=results)
    with open(os.path.join(out, 'summary.json'), 'w') as f: json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
