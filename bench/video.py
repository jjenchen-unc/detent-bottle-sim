"""Record a policy rollout as an mp4 (agentview | frontview | topview side by side, 320 px each).

python -m bench.video --policy bench.rollout:expert_policy --pose .02 -.01 5 --out runs/expert.mp4
python -m bench.video --policy bench.act.policy:make_policy --kwargs '{"ckpt": "runs/act/quick_l3/last.pt"}' --pose .02 -.01 5 --out runs/act.mp4
"""
import argparse
import json
import imageio.v2 as imageio
import numpy as np
from .evaluate import load
from .rollout import rollout

VIEWS = ('agentview', 'frontview', 'topview')


class Recorder:
    """Wraps a policy and renders the free cameras every `every` ticks."""

    def __init__(self, policy, every=2, size=320):
        self.policy, self.every, self.size, self.frames, self.t = policy, every, size, [], 0

    def reset(self, sim):
        if hasattr(self.policy, 'reset'): self.policy.reset(sim)

    def __call__(self, obs, sim):
        if self.t % self.every == 0:
            self.frames.append(np.concatenate([sim.render(v, self.size) for v in VIEWS], 1))
        self.t += 1
        return self.policy(obs, sim)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--policy', required=True); p.add_argument('--kwargs', default='{}')
    p.add_argument('--pose', nargs=3, type=float, default=[0., 0., 0.])
    p.add_argument('--max-steps', type=int, default=1200); p.add_argument('--out', required=True)
    args = p.parse_args()
    policy = load(args.policy)(args.pose, **json.loads(args.kwargs))
    rec = Recorder(policy)
    summary = rollout(rec, args.pose, cameras=getattr(policy, 'cameras', ()), image_size=getattr(policy, 'image_size', 128),
                      max_steps=args.max_steps)[0]
    imageio.mimsave(args.out, rec.frames, fps=10)
    print(summary, '->', args.out)
