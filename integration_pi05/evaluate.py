"""Reproducible scripted/hybrid episodes: python -m integration_pi05.evaluate."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import subprocess
import time

import numpy as np

from detent_bottle.gym_env import CAMERAS, DetentBottleEnv
from detent_bottle.policy import opening_sequence
from .actions import ActionAdapter, resample
from .observations import request
from .mechanism_response import format_task_prompt
from .client import CONTROLLED_POLICY, validate_expected_policy

DEFAULTS = {
    'checkpoint': 'gs://openpi-assets/checkpoints/pi05_droid',
    'config': 'pi05_droid', 'cameras': ['agentview', 'left_wrist'],
    'external_camera': 'agentview',
    'expected_server_instance': None,
    'image_size': 224, 'max_steps': 2400, 'execution_ticks': 4,
    'prompt': 'Fold the safety loop fully down, then press the rocker release to open the bottle. The bottle is held steady.',
    'host': 'localhost', 'port': 8000,
    'connection_timeout': 600, 'inference_timeout': 60, 'sampling_seed': None,
    'expected_policy': dict(CONTROLLED_POLICY),
    'mechanism_response': None,
}


def metadata():
    """Record client-side provenance; server provenance must be supplied separately."""
    def git(*args):
        result = subprocess.run(['git', *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    packages = {}
    for package in ('mujoco', 'numpy', 'scipy', 'gymnasium', 'Pillow', 'openpi-client'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': packages, 'simulator_revision': git('rev-parse', 'HEAD'),
            'working_tree_status': git('status', '--short'),
            'inference_rng_controlled': False, 'sampling_verified': False,
            'inference_rng_note': 'Client seed does not set the remote policy RNG. Attach server revision, checkpoint hash and RNG settings separately.'}


def load_settings(path=None, overrides=None):
    config = dict(DEFAULTS)
    if path:
        supplied = json.loads(Path(path).read_text())
        if not isinstance(supplied, dict):
            raise ValueError('Configuration must be a JSON object')
        unknown = supplied.keys() - DEFAULTS.keys()
        if unknown:
            raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
        config.update(supplied)
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    seed = config['sampling_seed']
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed >= 2**32):
        raise ValueError('sampling_seed must be null or an integer in [0, 2**32)')
    expected = config['expected_server_instance']
    if expected is not None and (not isinstance(expected, str) or not expected.strip() or seed is None):
        raise ValueError('expected_server_instance must be a nonempty string and requires controlled sampling_seed')
    for name in ('image_size', 'max_steps', 'execution_ticks', 'port'):
        if not isinstance(config[name], int) or isinstance(config[name], bool) or config[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if config['execution_ticks'] > 20:
        raise ValueError('execution_ticks cannot exceed the one-second action chunk')
    if config['port'] > 65535:
        raise ValueError('port cannot exceed 65535')
    for name in ('connection_timeout', 'inference_timeout'):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    expected_policy = validate_expected_policy(config['expected_policy'])
    config['expected_policy'] = expected_policy
    if (config['config'] != expected_policy['configuration'] or
            config['checkpoint'] != expected_policy['checkpoint']):
        raise ValueError('Evaluator config/checkpoint must exactly match the explicit expected_policy contract')
    if config['mechanism_response'] is not None:
        from .mechanism_response import validate_mechanism_response
        config['mechanism_response'] = list(validate_mechanism_response(config['mechanism_response']))
    if expected_policy != CONTROLLED_POLICY and config['sampling_seed'] is None:
        raise ValueError('A non-stock expected_policy requires explicit sampling_seed control')
    if config['image_size'] != 224:
        raise ValueError('pi05_droid requires image_size=224')
    cameras = config['cameras']
    if not isinstance(cameras, list) or not all(isinstance(c, str) and c in CAMERAS for c in cameras):
        raise ValueError(f'cameras must be a list of known camera names: {list(CAMERAS)}')
    if len(set(cameras)) != len(cameras):
        raise ValueError('cameras must not contain duplicates')
    external = config['external_camera']
    if external not in ('agentview', 'frontview', 'topview'):
        raise ValueError('external_camera must be agentview, frontview or topview')
    if not {external, 'left_wrist'}.issubset(cameras):
        raise ValueError('DROID observation adapter requires the selected external_camera and left_wrist')
    for name in ('host', 'prompt'):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f'{name} must be a nonempty string')
    return config


def load_episodes(path=None, seed=0, max_steps=2400):
    """Expand manifest conditions into concrete episodes without scene randomization."""
    conditions = json.loads(Path(path).read_text()) if path else [{'seeds': [seed]}]
    if not isinstance(conditions, list) or not conditions:
        raise ValueError('Episode manifest must be a nonempty list')
    episodes = []
    for condition in conditions:
        if not isinstance(condition, dict):
            raise ValueError('Each episode manifest condition must be a JSON object')
        unknown = condition.keys() - {'bottle_pose', 'seeds', 'max_steps', 'handoff_stage'}
        if unknown:
            raise ValueError(f'Unknown episode keys: {sorted(unknown)}')
        pose = condition.get('bottle_pose')
        if pose is not None and (not isinstance(pose, list) or len(pose) != 3 or
                                 not all(isinstance(n, (int, float)) and not isinstance(n, bool) and np.isfinite(n) for n in pose)):
            raise ValueError('bottle_pose must be finite [x, y, yaw_deg]')
        limit = condition.get('max_steps', max_steps)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError('Episode max_steps must be a positive integer')
        seeds = condition.get('seeds', [seed])
        if not isinstance(seeds, list) or not seeds or any(not isinstance(s, int) or isinstance(s, bool) or not 0 <= s < 2**32 for s in seeds):
            raise ValueError('seeds must be a nonempty list of integers in [0, 2**32)')
        for episode_seed in seeds:
            handoff_stage = condition.get('handoff_stage', 'before_left_arm')
            if handoff_stage not in ('before_left_arm', 'after_pinch'):
                raise ValueError("handoff_stage must be 'before_left_arm' or 'after_pinch'")
            episode = {'seed': episode_seed, 'bottle_pose': pose, 'max_steps': limit,
                       'handoff_stage': handoff_stage}
            episodes.append(episode)
    return episodes


def run_episode(out, controller, episode, config):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    random.seed(episode['seed']); np.random.seed(episode['seed'])
    start = time.monotonic()
    env = None; client = None; adapter = None
    preflight_latencies = []; preflight_server_metadata = {}; preflight_transport_events = []
    control_connection_verified = False
    control_expected_instance = config.get('expected_server_instance')
    rows = []; control_rows = []; error = None; failure_category = None; fingerprint = None
    request_index = 0; action_chunks_received = 0
    phase = 'initialization'; setup_complete = controller == 'scripted'; first_success_tick = None
    mechanism_response = config.get('mechanism_response')
    manifest = {'controller': controller, 'episode': episode, 'configuration': config, **metadata()}
    (out / 'run.json').write_text(json.dumps(manifest, indent=2))
    try:
        env = DetentBottleEnv(image_size=config['image_size'],
                             cameras=config['cameras'] if controller == 'hybrid' else (),
                             max_steps=episode['max_steps'], bottle_pose=episode['bottle_pose'])
        # Gym's reset seed controls its RNG; simulator reset itself is deterministic.
        env.reset(seed=episode['seed'])
        s = env.sim; adapter = ActionAdapter(s)
        fingerprint = hashlib.sha256(s.data.qpos.tobytes()).hexdigest()
        np.savez_compressed(out / 'initial.npz', qpos=s.data.qpos.copy(), qvel=s.data.qvel.copy())

        def record_state():
            nonlocal first_success_tick
            if s.success and first_success_tick is None:
                first_success_tick = s.steps
            rows.append(dict(qpos=s.data.qpos.copy(), qvel=s.data.qvel.copy(),
                             targets=s.targets.copy(), rotations=s.rotations.copy(),
                             grips=s.grip_targets.copy(), eef_pos=s.data.site_xpos[s.tips].copy(),
                             eef_rotation=s.data.site_xmat[s.tips].reshape(2, 3, 3).copy(),
                             bottle_qpos=s.bottle_qpos(), wrist_ft=s.wrenches_world(),
                             step=s.steps, sim_time=s.data.time, **s.mechanism()))

        def step():
            s.step()
            record_state()

        handoff_stage = episode.get('handoff_stage', 'before_left_arm')
        if handoff_stage == 'after_pinch' and controller != 'hybrid':
            raise ValueError('after_pinch handoff is supported only for hybrid VLA episodes')
        effective_prompt = config['prompt']
        if mechanism_response is not None:
            effective_prompt = format_task_prompt(config['prompt'], mechanism_response)
            manifest['effective_prompt'] = effective_prompt
            (out / 'run.json').write_text(json.dumps(manifest, indent=2))
        if controller == 'hybrid' and config.get('sampling_seed') is not None and handoff_stage != 'after_pinch':
            phase = 'inference'
            from .client import Client
            from .preflight import run_preflight
            preflight_client = Client(config['host'], config['port'], timeout=config['inference_timeout'],
                                      connection_timeout=config['connection_timeout'])
            try:
                run_preflight(preflight_client, request(env._obs(), effective_prompt, config['external_camera']), out / 'preflight',
                              config['sampling_seed'], expected_instance=config['expected_server_instance'],
                              expected_policy=config['expected_policy'])
            finally:
                preflight_latencies = list(preflight_client.latencies)
                preflight_server_metadata = dict(preflight_client.server_metadata)
                preflight_transport_events = list(getattr(preflight_client, 'transport_events', []))
                preflight_client.close()
            if control_expected_instance is None:
                control_expected_instance = preflight_server_metadata.get('server_instance_id')
            manifest['inference_rng_controlled'] = True
            manifest['sampling_verified'] = True
            manifest['inference_rng_note'] = 'Explicit request noise; stationary A/B/A preflight passed. Simulation seed is separate.'
            (out / 'run.json').write_text(json.dumps(manifest, indent=2))
        phase = 'setup' if controller == 'hybrid' else 'scripted'
        for stage, arm, pos, rpy, gap, ticks in opening_sequence():
            if handoff_stage == 'after_pinch' and stage.startswith('3 / Fold'):
                setup_complete = True
                manifest['effective_prompt'] = effective_prompt
                (out / 'run.json').write_text(json.dumps(manifest, indent=2))
                break
            if controller == 'hybrid' and arm == 0 and handoff_stage == 'before_left_arm':
                setup_complete = True
                break
            s.target(arm, pos, rpy, gap)
            for _ in range(ticks):
                step()
                if s.reason:
                    break
            if s.reason:
                break
        if controller == 'hybrid' and not s.reason and handoff_stage == 'after_pinch' and config.get('sampling_seed') is not None:
            phase = 'inference'
            from .client import Client
            from .preflight import run_preflight
            preflight_client = Client(config['host'], config['port'], timeout=config['inference_timeout'],
                                      connection_timeout=config['connection_timeout'])
            try:
                run_preflight(preflight_client,
                              request(env._obs(), effective_prompt, config['external_camera']),
                              out / 'preflight', config['sampling_seed'],
                              expected_instance=config['expected_server_instance'],
                              expected_policy=config.get('expected_policy'))
            finally:
                preflight_latencies = list(preflight_client.latencies)
                preflight_server_metadata = dict(preflight_client.server_metadata)
                preflight_transport_events = list(getattr(preflight_client, 'transport_events', []))
                preflight_client.close()
            if control_expected_instance is None:
                control_expected_instance = preflight_server_metadata.get('server_instance_id')
            manifest['inference_rng_controlled'] = True
            manifest['sampling_verified'] = True
            manifest['effective_prompt'] = effective_prompt
            (out / 'run.json').write_text(json.dumps(manifest, indent=2))
        if controller == 'hybrid' and not s.reason:
            phase = 'inference'
            from .client import Client
            if client is None:
                client = Client(config['host'], config['port'], timeout=config['inference_timeout'],
                                connection_timeout=config['connection_timeout'])
            if config.get('sampling_seed') is not None:
                if control_expected_instance is None:
                    client.connect(controlled=True, expected_policy=config['expected_policy'])
                else:
                    client.connect(controlled=True, expected_instance=control_expected_instance,
                                    expected_policy=config['expected_policy'])
                    if client.server_metadata.get('server_instance_id') != control_expected_instance:
                        raise ValueError('Control server instance changed after scripted setup')
                control_connection_verified = True
            while not s.reason:
                phase = 'inference'
                obs = env._obs()
                payload = request(obs, effective_prompt, config['external_camera'])
                sampling = {}
                if config.get('sampling_seed') is not None:
                    from .preflight import sampled_request
                    payload, sampling = sampled_request(payload, config['sampling_seed'], request_index)
                # Include exact RGB and robot-state request before networking, including failed calls.
                np.savez_compressed(out / f'request-{s.steps}.npz',
                                    **{k: v for k, v in payload.items() if k != 'prompt'})
                (out / f'request-{s.steps}.json').write_text(json.dumps({'step': s.steps, 'prompt': payload['prompt'], **sampling}))
                chunk = client.infer(payload)
                request_index += 1
                action_chunks_received += 1
                np.save(out / f'chunk-{s.steps}.npy', chunk)
                phase = 'control'
                request_step = s.steps
                for chunk_tick, cmd in enumerate(resample(chunk, config['execution_ticks'])):
                    weights = np.array([max(0, min((chunk_tick + 1) / 20, (j + 1) / 15)
                                            - max(chunk_tick / 20, j / 15)) * 15 for j in range(15)])
                    record = dict(request_step=request_step, chunk_tick=chunk_tick,
                                  pre_step=s.steps, post_step=-1, step_completed=False,
                                  source_weights=weights, source_action_clipped=np.any(np.abs(chunk[:, :7]) > 1, axis=1),
                                  command=cmd.copy(),
                                  pre_eef_pos=s.data.site_xpos[s.tips[0]].copy(),
                                  pre_eef_rotation=s.data.site_xmat[s.tips[0]].reshape(3, 3).copy(),
                                  post_eef_pos=None, post_eef_rotation=None,
                                  measured_closure=None, wrist_ft=None)
                    # Append before control so a failed apply/step remains visible.
                    control_rows.append(record)
                    adapter.apply(cmd, fraction=1)
                    record.update(adapter.last_diagnostics)
                    step()
                    record.update(post_step=s.steps, step_completed=True,
                                  post_eef_pos=s.data.site_xpos[s.tips[0]].copy(),
                                  post_eef_rotation=s.data.site_xmat[s.tips[0]].reshape(3, 3).copy(),
                                  measured_closure=1 - s.openings()[0] / .09,
                                  wrist_ft=s.wrenches_world().copy())
                    if s.reason:
                        break
        if controller == 'hybrid' and not setup_complete:
            failure_category = 'setup_failure'
        elif not s.success:
            failure_category = 'task_failure'
    except Exception as exc:
        error = repr(exc)
        failure_category = 'infrastructure_failure' if phase in ('initialization', 'inference') else ('setup_failure' if phase == 'setup' else 'control_failure')
    finally:
        np.savez_compressed(out / 'trajectory.npz', **{k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else {})
        # JSONL supports incomplete records when apply fails before diagnostics exist.
        with (out / 'control.jsonl').open('w') as stream:
            for record in control_rows:
                stream.write(json.dumps(record, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()) + '\n')
        sim = env.sim if env else None
        final = {'controller': controller, **episode, 'success': bool(sim.success) if sim else False,
                 'reason': sim.reason if sim else None, 'error': error,
                 'failure_category': failure_category, 'phase': phase, 'setup_complete': setup_complete,
                 'first_success_tick': first_success_tick, 'steps': sim.steps if sim else 0,
                 'sim_s': sim.data.time if sim else 0, 'wall_s': time.monotonic() - start,
                 'initial_fingerprint': fingerprint, 'workspace_clips': adapter.clips if adapter else 0,
                 'latencies': preflight_latencies + (list(client.latencies) if client else [])}
        final['experiment_kind'] = ('bounded_motion_diagnostic' if controller == 'hybrid' and episode['max_steps'] == 395 else 'episode')
        final['learned_ticks'] = sum(r['step_completed'] for r in control_rows)
        final['inference_requests_completed'] = request_index
        final['policy_action_chunks_received'] = action_chunks_received
        final['simulator_action_ticks_completed'] = final['learned_ticks']
        final['transport_events'] = list(getattr(client, 'transport_events', [])) if client else []
        final['diagnostic_limit_reached'] = bool(final['experiment_kind'] == 'bounded_motion_diagnostic' and final['reason'] == 'time_limit' and setup_complete and not error)
        final['preflight_latencies'] = preflight_latencies
        final['preflight_transport_events'] = preflight_transport_events
        final['control_latencies'] = list(client.latencies) if client else []
        final['preflight_server_metadata'] = preflight_server_metadata
        final['control_server_metadata'] = dict(client.server_metadata) if client else {}
        final['server_metadata'] = final['control_server_metadata'] or preflight_server_metadata
        final['control_connection_verified'] = control_connection_verified
        final['control_expected_instance'] = control_expected_instance
        final['expected_policy'] = config.get('expected_policy')
        final['mechanism_response'] = mechanism_response
        final['sampling_verified'] = manifest['sampling_verified']
        final['sampling_seed'] = config.get('sampling_seed')
        (out / 'final.json').write_text(json.dumps(final, indent=2))
        if client:
            client.close()
        if env:
            env.close()
    return final


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller', choices=['scripted', 'hybrid'], default='scripted')
    p.add_argument('--out', required=True)
    p.add_argument('--config', help='JSON configuration; CLI values override it')
    p.add_argument('--episodes', help='JSON list of pose/seed conditions; writes one subdirectory per expanded episode')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--host'); p.add_argument('--port', type=int)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--execution-ticks', type=int)
    p.add_argument('--sampling-seed', type=int, help='Explicit-noise sampling seed, separate from simulator seed')
    p.add_argument('--expected-instance', dest='expected_server_instance', help='Require this persistent server instance before inference or motion')
    args = p.parse_args()
    try:
        config = load_settings(args.config, {k: getattr(args, k) for k in ('host', 'port', 'max_steps', 'execution_ticks', 'sampling_seed', 'expected_server_instance')})
        episodes = load_episodes(args.episodes, args.seed, config['max_steps'])
    except (ValueError, TypeError) as exc:
        p.error(str(exc))
    root = Path(args.out)
    if args.episodes:
        root.mkdir(parents=True, exist_ok=False)
        results = [run_episode(root / f'episode-{i:04d}-seed-{episode["seed"]}', args.controller, episode, config)
                   for i, episode in enumerate(episodes)]
        (root / 'results.json').write_text(json.dumps(results, indent=2))
    else:
        results = [run_episode(root, args.controller, episodes[0], config)]
    print(json.dumps(results, indent=2))
    if any(result['error'] for result in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
