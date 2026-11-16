"""Inference without simulation stepping; run from the installed simulator environment."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from integration_pi05.client import Client
from integration_pi05.evaluate import metadata
from integration_pi05.observations import request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--out', required=True, help='New artifact directory')
    parser.add_argument('--synthetic', action='store_true', help='Use random RGB/state instead of rendering')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--connection-timeout', type=float, default=600)
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    env = None; error = None; result = None
    start = time.monotonic()
    client = Client(args.host, args.port, timeout=args.timeout, connection_timeout=args.connection_timeout)
    try:
        if args.synthetic:
            rng = np.random.default_rng(args.seed)
            payload = {'observation/exterior_image_1_left': rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
                       'observation/wrist_image_left': rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
                       'observation/joint_position': rng.random(7),
                       'observation/gripper_position': rng.random(1), 'prompt': 'Open the bottle'}
        else:
            from detent_bottle.gym_env import DetentBottleEnv
            env = DetentBottleEnv(image_size=224, cameras=('agentview', 'left_wrist'))
            obs, _ = env.reset(seed=args.seed)
            initial = env.sim.data.qpos.copy()
            payload = request(obs, 'Open the bottle')
        np.savez_compressed(out / 'request.npz', **{k: v for k, v in payload.items() if k != 'prompt'})
        (out / 'prompt.txt').write_text(payload['prompt'])
        result = client.infer(payload)
        np.save(out / 'actions.npy', result)
        if env is not None:
            if env.sim.steps != 0 or not np.array_equal(initial, env.sim.data.qpos):
                raise RuntimeError('Smoke test unexpectedly changed simulation state')
    except Exception as exc:
        error = repr(exc)
    finally:
        client.close()
        if env is not None: env.close()
        report = {**vars(args), **metadata(), 'error': error,
                  'shape': list(result.shape) if result is not None else None,
                  'finite': bool(np.isfinite(result).all()) if result is not None else False,
                  'latencies': client.latencies, 'wall_s': time.monotonic() - start}
        report['server_metadata'] = client.server_metadata
        (out / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if error: raise SystemExit(1)


if __name__ == '__main__':
    main()
