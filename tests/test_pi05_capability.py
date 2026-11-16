import json
from pathlib import Path

import pytest

from integration_pi05 import capability
from integration_pi05.client import CONTROLLED_POLICY


def inputs(tmp_path):
    contract = {**CONTROLLED_POLICY, 'checkpoint': '/candidate/9999'}
    config = {
        'checkpoint': contract['checkpoint'],
        'expected_policy': contract,
        'max_steps': 2400,
        'sampling_seed': 0,
        'execution_ticks': 4,
        'external_camera': 'frontview',
        'cameras': ['frontview', 'left_wrist'],
        'mechanism_response': [.1935, .6268],
    }
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    final = {
        'success': False,
        'error': None,
        'sampling_verified': True,
        'control_connection_verified': True,
        'learned_ticks': 4,
        'policy_action_chunks_received': 1,
        'steps': 500,
        'reason': 'time_limit',
        'mechanism_response': config['mechanism_response'],
        'server_metadata': {
            **contract,
            'server_instance_id': 'instance',
            'finetune_mode': 'lora',
            'checkpoint_lora_tensors': 20,
        },
    }
    return path, final


@pytest.mark.parametrize('success', [False, True])
def test_nominal_gate_uses_post_pinch_no_probe_and_full_horizon(tmp_path, monkeypatch, success):
    config, final = inputs(tmp_path)
    final['success'] = success

    def episode(out, controller, condition, settings):
        assert controller == 'hybrid'
        assert condition['handoff_stage'] == 'after_pinch'
        assert condition['max_steps'] == 2400 and condition['seed'] == 0
        assert settings['expected_server_instance'] == 'instance'
        Path(out).mkdir()
        return final

    monkeypatch.setattr(capability, 'run_episode', episode)
    result = capability.run_capability(tmp_path / 'episode', config, 'instance')
    assert result['pipeline_passed']
    assert result['success'] == success
    assert result['outcome'] == ('task_success' if success else 'policy_outcome_failure')


@pytest.mark.parametrize('change', [
    {'error': 'handshake timeout', 'learned_ticks': 0},
    {'mechanism_response': [1, 0]},
    {'policy_action_chunks_received': 0},
    {'control_connection_verified': False},
])
def test_pipeline_errors_cannot_count_as_policy_outcomes(tmp_path, monkeypatch, change):
    config, final = inputs(tmp_path)
    final.update(change)
    monkeypatch.setattr(capability, 'run_episode', lambda out, *_: (Path(out).mkdir(), final)[1])
    result = capability.run_capability(tmp_path / 'episode', config, 'instance')
    assert not result['pipeline_passed']
    assert result['outcome'] == 'setup_or_infrastructure_failure'


@pytest.mark.parametrize('change', [
    {'max_steps': 700},
    {'sampling_seed': None},
])
def test_non_nominal_gate_settings_are_rejected_before_motion(tmp_path, monkeypatch, change):
    config, _ = inputs(tmp_path)
    values = json.loads(config.read_text())
    values.update(change)
    config.write_text(json.dumps(values))
    monkeypatch.setattr(capability, 'run_episode', lambda *_: pytest.fail('motion must not start'))
    with pytest.raises(ValueError):
        capability.run_capability(tmp_path / 'episode', config, 'instance')
