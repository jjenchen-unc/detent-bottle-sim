import json

import pytest

from integration_pi05.summarize import summarize, wilson


def save(root, name, **updates):
    path = root / name / 'final.json'
    path.parent.mkdir()
    record = dict(controller='hybrid', success=False, steps=2400, reason='time_limit',
                  error=None, setup_complete=True, bottle_pose=[0, 0, 0], wall_s=2)
    record.update(updates)
    path.write_text(json.dumps(record))
    return path


def test_denominators_and_setup_failures(tmp_path):
    save(tmp_path, 'success', success=True, reason='success', steps=715, first_success_tick=715)
    save(tmp_path, 'failure')
    save(tmp_path, 'setup', steps=20, reason='bottle_fell', setup_complete=False, failure_category='setup_failure')
    save(tmp_path, 'network', steps=295, error='timeout', failure_category='infrastructure_failure')
    save(tmp_path, 'control', steps=300, failure_category='control_failure')
    result = summarize([tmp_path])['tracks']['hybrid']
    assert result['attempts'] == 5
    assert result['completed_task_attempts'] == 3
    assert result['excluded_error_attempts'] == 2
    assert result['successes'] == 1
    assert result['setup_failures'] == 1
    assert result['success_rate'] == 1 / 3
    assert result['first_success_sim_seconds']['mean'] == 35.75
    assert result['all_attempt_steps']['count'] == 5
    assert result['successful_episode_steps']['mean'] == 715
    assert len(result['recorded_pose_counts']) == 1


def test_errors_only_and_missing_first_success(tmp_path):
    save(tmp_path, 'error', error='broken pipe')
    summary = summarize([tmp_path])['tracks']['hybrid']
    assert summary['wilson_95'] is None
    assert summary['success_rate'] is None
    assert summary['first_success_ticks']['count'] == 0
    save(tmp_path, 'legacy', success=True, steps=715)
    summary = summarize([tmp_path])['tracks']['hybrid']
    assert summary['successes_missing_first_success_tick'] == 1
    assert summary['first_success_ticks']['count'] == 0


def test_bounded_diagnostics_are_separate_from_task_denominator(tmp_path):
    save(tmp_path, 'full-success', success=True, reason='success', steps=715)
    save(tmp_path, 'kind-marker', success=False, steps=395,
         experiment_kind='bounded_motion_diagnostic')
    save(tmp_path, 'flag-marker', success=True, steps=395,
         diagnostic_limit_reached=True)

    tracks = summarize([tmp_path])['tracks']
    assert tracks['hybrid']['attempts'] == 1
    assert tracks['hybrid']['completed_task_attempts'] == 1
    assert tracks['hybrid']['successes'] == 1
    assert tracks['hybrid']['success_rate'] == 1
    diagnostic = tracks['hybrid:bounded_motion_diagnostic']
    assert diagnostic['attempts'] == 2
    assert diagnostic['bounded_diagnostic_attempts'] == 2
    assert diagnostic['completed_task_attempts'] == 0
    assert diagnostic['successes'] == 0
    assert diagnostic['success_rate'] is None
    assert diagnostic['wilson_95'] is None


def test_llm_separate_and_invalid_results(tmp_path):
    save(tmp_path, 'script', controller='scripted')
    llm = save(tmp_path, 'llm', controller=None, scaffold='v3', mode='live', privileged=False, sensing='both')
    save(tmp_path, 'dry', controller=None, scaffold='v3', mode='dry', privileged=False, sensing='both')
    bad = tmp_path / 'bad'; bad.mkdir(); (bad / 'final.json').write_text('not json')
    result = summarize([tmp_path, llm])
    assert result['discovered_result_files'] == 4  # deduplicate overlapping roots
    assert len(result['tracks']) == 3
    assert len(result['invalid_results']) == 1
    assert 'scripted' in result['tracks']
    assert all(name.startswith('llm:') for name in result['tracks'] if name != 'scripted')


def test_wilson_known_values():
    assert wilson(0, 0) is None
    low, high = wilson(5, 10)
    assert low == pytest.approx(.23659309)
    assert high == pytest.approx(.76340691)
    assert wilson(0, 10)[0] == pytest.approx(0)
    assert wilson(10, 10)[1] == pytest.approx(1)
