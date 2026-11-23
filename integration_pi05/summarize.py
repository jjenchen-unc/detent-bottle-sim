"""Summarize final.json artifacts without treating repeat seeds as new scenes."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


def wilson(successes, count, z=1.959963984540054):
    """95% binomial Wilson interval; undefined when no completed attempts exist."""
    if count == 0:
        return None
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def distribution(values):
    values = list(values)
    if not values:
        return {'count': 0, 'mean': None, 'median': None, 'min': None, 'max': None}
    return {'count': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'min': min(values), 'max': max(values)}


def _track(record):
    controller = record.get('controller')
    if controller in ('scripted', 'hybrid'):
        return controller
    if 'scaffold' in record and 'mode' in record:
        # LLM scaffold/sensing/privileged/dry settings must never pool with VLA.
        return 'llm:' + json.dumps({k: record.get(k) for k in
                                    ('scaffold', 'mode', 'sensing', 'privileged')}, sort_keys=True)
    raise ValueError('Unrecognized controller/LLM result schema')


def _finite_nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def summarize(roots):
    files = set()
    for root in roots:
        root = Path(root)
        if root.is_file():
            files.add(root.resolve())
        elif root.is_dir():
            files.update(p.resolve() for p in root.rglob('final.json'))
        else:
            raise ValueError(f'Input path does not exist: {root}')
    groups = {}; invalid = []
    for path in sorted(files):
        try:
            record = json.loads(path.read_text())
            if not isinstance(record, dict) or not isinstance(record.get('success'), bool):
                raise ValueError('Result must contain boolean success')
            if not _finite_nonnegative(record.get('steps')):
                raise ValueError('Result must contain finite nonnegative steps')
            name = _track(record)
            if (record.get('experiment_kind') == 'bounded_motion_diagnostic' or
                    record.get('diagnostic_limit_reached') is True):
                name += ':bounded_motion_diagnostic'
            groups.setdefault(name, []).append((path, record))
        except (ValueError, OSError, TypeError) as exc:
            invalid.append({'path': str(path), 'error': str(exc)})
    tracks = {}
    for name, rows in sorted(groups.items()):
        bounded_diagnostic = name.endswith(':bounded_motion_diagnostic')
        excluded = [(p, r) for p, r in rows if r.get('error') or r.get('failure_category') in
                    ('infrastructure_failure', 'control_failure') or r.get('reason') in
                    ('provider_error', 'infrastructure_failure', 'control_failure')]
        excluded_paths = {p for p, _ in excluded}
        completed = [] if bounded_diagnostic else [(p, r) for p, r in rows if p not in excluded_paths]
        successful = [(p, r) for p, r in completed if r['success']]
        # Legacy canonical runs have no pose field; preserve that uncertainty.
        poses = Counter(json.dumps(r.get('bottle_pose', 'unrecorded'), sort_keys=True) for _, r in rows)
        first_ticks = [r.get('first_success_tick') for _, r in successful]
        recorded_first = [v for v in first_ticks if _finite_nonnegative(v)]
        tracks[name] = {
            'attempts': len(rows),
            'bounded_diagnostic_attempts': len(rows) if bounded_diagnostic else 0,
            'completed_task_attempts': len(completed),
            'excluded_error_attempts': len(excluded), 'successes': len(successful),
            'success_rate': len(successful) / len(completed) if completed else None,
            'wilson_95': wilson(len(successful), len(completed)),
            'setup_failures': sum(r.get('failure_category') == 'setup_failure' or
                                  (r.get('controller') == 'hybrid' and r.get('setup_complete') is False and r.get('phase') == 'setup')
                                  for _, r in rows),
            'termination_reasons': dict(Counter(str(r.get('reason') or 'unspecified') for _, r in completed)),
            'failure_categories': dict(Counter(str(r.get('failure_category') or 'none') for _, r in rows)),
            'all_attempt_steps': distribution(r['steps'] for _, r in rows),
            'completed_attempt_steps': distribution(r['steps'] for _, r in completed),
            'successful_episode_steps': distribution(r['steps'] for _, r in successful),
            'first_success_ticks': distribution(recorded_first),
            'first_success_sim_seconds': distribution(v * .05 for v in recorded_first),
            'successes_missing_first_success_tick': len(successful) - len(recorded_first),
            'wall_seconds': distribution(r['wall_s'] for _, r in rows if _finite_nonnegative(r.get('wall_s'))),
            'recorded_pose_counts': dict(poses),
            'errors': [{'path': str(p), 'error': r.get('error'),
                        'category': r.get('failure_category'), 'reason': r.get('reason')} for p, r in excluded],
            'source_files': [str(p) for p, _ in rows],
        }
    return {'discovered_result_files': len(files), 'invalid_results': invalid, 'tracks': tracks,
            'interpretation': 'Repeated seeds at the same pose are repeated trials of one scene, not independent scene samples. Wilson intervals describe trial outcomes and do not establish generalization. LLM tracks use different sensing and control scaffolds. Unrecorded first-success times are not inferred from episode end.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', help='Result directories or explicit final.json files')
    parser.add_argument('--out', required=True, help='Destination summary JSON (must not exist)')
    args = parser.parse_args()
    try:
        result = summarize(args.inputs)
        # Avoid silently destroying an existing report or input artifact.
        with Path(args.out).open('x') as output:
            json.dump(result, output, indent=2)
            output.write('\n')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({'tracks': list(result['tracks']), 'result_files': result['discovered_result_files'],
                      'invalid_results': len(result['invalid_results'])}))


if __name__ == '__main__':
    main()
