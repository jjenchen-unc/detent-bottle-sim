"""Stationary A/B/A explicit-noise preflight; never steps simulation."""
import argparse
import json
from pathlib import Path

import numpy as np
from .client import validate_expected_policy
from .sampling import NOISE_KEY, make_noise, noise_hash, sampling_metadata


def sampled_request(payload, seed, index):
    noise = make_noise(seed, index)
    return {**payload, NOISE_KEY: noise}, {'sampling_seed':seed, 'request_index':index,
                                         'noise_sha256':noise_hash(noise), **sampling_metadata()}


def save_request(out, name, payload, provenance):
    """Persist exact numeric request and sampling metadata before networking."""
    out = Path(out)
    np.savez_compressed(out / f'{name}.npz', **{k:v for k,v in payload.items() if k != 'prompt'})
    (out / f'{name}.json').write_text(json.dumps({'prompt':payload['prompt'], **provenance}, indent=2))


def run_preflight(client, payload, out, sampling_seed, expected_instance=None, expected_policy=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    result = {'passed':False, 'sampling_seed':sampling_seed, 'sequence':['A','B','A'],
              'atol':1e-6, 'rtol':0, 'error':None, 'expected_instance_id':expected_instance}
    actions = []
    try:
        result['expected_policy'] = validate_expected_policy(expected_policy)
        connection = {'controlled': True}
        if expected_instance is not None:
            connection['expected_instance'] = expected_instance
        if expected_policy is not None:
            connection['expected_policy'] = result['expected_policy']
        client.connect(**connection)
        result['expected_instance_id'] = expected_instance
        result['server_metadata'] = client.server_metadata
        for name, index in (('A-first',0), ('B',1), ('A-repeat',0)):
            sampled, provenance = sampled_request(payload, sampling_seed, index)
            save_request(out, name, sampled, provenance)
            action = client.infer(sampled)
            np.save(out / f'{name}-actions.npy', action)
            actions.append(action)
        difference = np.abs(actions[0] - actions[2])
        result['max_abs_difference'] = float(difference.max())
        result['a_exact_equal'] = bool(np.array_equal(actions[0], actions[2]))
        result['b_vs_a_max_abs_difference'] = float(np.max(np.abs(actions[1] - actions[0])))
        result['passed'] = bool(np.allclose(actions[0], actions[2], atol=1e-6, rtol=0))
        if not result['passed']:
            raise RuntimeError('Stationary A/B/A explicit-noise preflight failed')
    except Exception as exc:
        result['error'] = repr(exc)
        raise
    finally:
        result.setdefault('server_metadata', dict(client.server_metadata))
        result['latencies'] = list(client.latencies)
        result['transport_events'] = list(getattr(client, 'transport_events', []))
        (out / 'result.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    from .client import Client
    from .evaluate import load_settings
    from .observations import request
    from detent_bottle.gym_env import DetentBottleEnv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config'); parser.add_argument('--sampling-seed', type=int)
    parser.add_argument('--host'); parser.add_argument('--port', type=int)
    parser.add_argument('--expected-instance')
    args = parser.parse_args()
    config = load_settings(args.config, {'host':args.host,'port':args.port,'sampling_seed':args.sampling_seed,
                                        'expected_server_instance':args.expected_instance})
    if config['sampling_seed'] is None:
        parser.error('sampling_seed must be set in --config or through --sampling-seed')
    env = DetentBottleEnv(image_size=224, cameras=config['cameras'])
    client = Client(config['host'], config['port'], timeout=config['inference_timeout'], connection_timeout=config['connection_timeout'])
    try:
        obs, _ = env.reset(seed=0)
        initial = env.sim.data.qpos.copy()
        report = run_preflight(client, request(obs, config['prompt'], config['external_camera']), args.out,
                               config['sampling_seed'], expected_instance=config['expected_server_instance'],
                               expected_policy=config.get('expected_policy'))
        if env.sim.steps != 0 or not np.array_equal(initial, env.sim.data.qpos):
            raise RuntimeError('Preflight changed simulator state')
        print(json.dumps(report, indent=2))
    finally:
        client.close(); env.close()


if __name__ == '__main__': main()
