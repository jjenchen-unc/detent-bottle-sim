"""Audit one nominal no-probe episode against a loaded π₀.₅ policy contract."""
import argparse
import json
from pathlib import Path

from .client import validate_expected_policy
from .evaluate import load_settings, run_episode


def run_capability(out, config_path, expected_instance, *, host=None, port=None, seed=0):
    if not isinstance(expected_instance, str) or not expected_instance.strip():
        raise ValueError('An exact server instance is required')
    if seed != 0:
        raise ValueError('The nominal capability gate uses simulator seed zero')
    config = load_settings(config_path, {'host': host, 'port': port,
                                         'expected_server_instance': expected_instance})
    required = {'max_steps': 2400, 'execution_ticks': 4,
                'sampling_seed': 0, 'external_camera': 'frontview',
                'image_size': 224}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError(f'Nominal capability gate requires {required}')
    contract = validate_expected_policy(config.get('expected_policy'))
    if contract['checkpoint'] != config['checkpoint']:
        raise ValueError('Candidate checkpoint and policy contract disagree')
    episode = {'seed': 0, 'bottle_pose': None, 'max_steps': 2400,
               'handoff_stage': 'after_pinch'}
    final = run_episode(out, 'hybrid', episode, config)
    metadata = final.get('server_metadata', {})
    descriptor_matches = final.get('mechanism_response') == config.get('mechanism_response')
    adapters_loaded = (metadata.get('finetune_mode') == 'lora' and
                       metadata.get('checkpoint_lora_tensors', 0) > 0)
    pipeline_passed = bool(
        not final.get('error') and final.get('sampling_verified') and
        final.get('control_connection_verified') and final.get('learned_ticks', 0) > 0 and
        final.get('policy_action_chunks_received', 0) > 0 and
        descriptor_matches and
        metadata.get('server_instance_id') == expected_instance and adapters_loaded and
        all(metadata.get(key) == value for key, value in contract.items()))
    result = {
        'pipeline_passed': pipeline_passed,
        'success': bool(final.get('success')),
        'outcome': ('setup_or_infrastructure_failure' if not pipeline_passed else
                    'task_success' if final.get('success') else 'policy_outcome_failure'),
        'checkpoint': contract['checkpoint'],
        'server_instance': expected_instance,
        'handoff_stage': 'after_pinch',
        'mechanism_response': final.get('mechanism_response'),
        'mechanism_response_matches_config': descriptor_matches,
        'steps': final.get('steps'),
        'learned_ticks': final.get('learned_ticks'),
        'reason': final.get('reason'),
        'error': final.get('error'),
        'failure_category': final.get('failure_category'),
        'scope': 'One nominal feasibility episode; no success-rate claim',
    }
    (Path(out) / 'capability-result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--expected-instance', required=True)
    parser.add_argument('--host')
    parser.add_argument('--port', type=int)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    result = run_capability(args.out, args.config, args.expected_instance,
                            host=args.host, port=args.port, seed=args.seed)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['pipeline_passed'] else 1)


if __name__ == '__main__':
    main()
