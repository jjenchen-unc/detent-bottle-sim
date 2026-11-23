import json
import numpy as np
import pytest

from integration_pi05.evaluate import load_settings, load_episodes, run_episode


def test_manifest_expands_conditions_without_randomizing(tmp_path):
    manifest = tmp_path / 'episodes.json'
    manifest.write_text(json.dumps([{'bottle_pose': [0, .01, 5], 'seeds': [0, 42],
                                     'max_steps': 3, 'handoff_stage': 'after_pinch'}]))
    episodes = load_episodes(manifest)
    assert episodes == [
        {'seed': 0, 'bottle_pose': [0, .01, 5], 'max_steps': 3, 'handoff_stage': 'after_pinch'},
        {'seed': 42, 'bottle_pose': [0, .01, 5], 'max_steps': 3, 'handoff_stage': 'after_pinch'},
    ]


@pytest.mark.parametrize('override', [
    {'execution_ticks': 21}, {'image_size': 128}, {'config': 'pi05_libero'},
    {'port': 65536}, {'inference_timeout': 0},
])
def test_reject_incompatible_settings(override):
    with pytest.raises(ValueError):
        load_settings(overrides=override)


def test_hybrid_handoff_after_pinch_uses_static_descriptor_without_probe(tmp_path, monkeypatch):
    import integration_pi05.client as client_module
    import integration_pi05.evaluate as evaluator

    class FakeClient:
        latencies = []
        server_metadata = {}
        transport_events = []

        def __init__(self, *args, **kwargs):
            pass

        def infer(self, payload):
            assert 'loop release excursion 0.2000' in payload['prompt']
            return np.zeros((15, 8))

        def close(self):
            pass

    monkeypatch.setattr(client_module, 'Client', FakeClient)
    monkeypatch.setattr(evaluator, 'opening_sequence', lambda: iter([
        ('hold bottle', 1, [.1, .1, 1.], [90, 0, 0], .05, 1),
        ('reach loop', 0, [0, -.05, 1.13], [180, 0, 0], .035, 1),
        ('pinch loop', 0, [0, -.049, 1.066], [180, 0, 0], .003, 1),
        ('3 / Fold loop', 0, [0, -.049, 1.066], [180, 0, 0], .003, 1),
    ]))

    image = np.zeros((224, 224, 3), dtype=np.uint8)
    monkeypatch.setattr('detent_bottle.gym_env.DetentBottleEnv._obs', lambda self: {
        'frontview': image, 'left_wrist': image, 'joint_pos': np.zeros(14),
        'gripper_qpos': np.full(4, .045),
    })
    config = load_settings(overrides={
        'cameras': ['frontview', 'left_wrist'],
        'external_camera': 'frontview',
        'mechanism_response': [.2, .6],
    })
    final = run_episode(tmp_path / 'episode', 'hybrid', {
        'seed': 0, 'bottle_pose': None, 'max_steps': 4, 'handoff_stage': 'after_pinch',
    }, config)
    assert final['setup_complete']
    assert final['mechanism_response'] == [.2, .6]
    assert final['learned_ticks'] == 1
    run = json.loads((tmp_path / 'episode' / 'run.json').read_text())
    assert 'Mechanism response' in run['effective_prompt']


def test_hybrid_handoff_before_left_arm_sends_static_descriptor(tmp_path, monkeypatch):
    import integration_pi05.client as client_module
    import integration_pi05.evaluate as evaluator
    from integration_pi05.mechanism_response import format_mechanism_response

    outgoing = []

    class FakeClient:
        latencies = []
        server_metadata = {}
        transport_events = []

        def __init__(self, *args, **kwargs):
            pass

        def infer(self, payload):
            outgoing.append(payload)
            return np.zeros((15, 8))

        def close(self):
            pass

    monkeypatch.setattr(client_module, 'Client', FakeClient)
    monkeypatch.setattr(evaluator, 'opening_sequence', lambda: iter([
        ('hold bottle', 1, [.1, .1, 1.], [90, 0, 0], .05, 1),
        ('reach loop', 0, [0, -.05, 1.13], [180, 0, 0], .035, 1),
    ]))
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    monkeypatch.setattr('detent_bottle.gym_env.DetentBottleEnv._obs', lambda self: {
        'agentview': image, 'left_wrist': image, 'joint_pos': np.zeros(14),
        'gripper_qpos': np.full(4, .045),
    })
    config = load_settings(overrides={'mechanism_response': [.2, .6]})
    run_episode(tmp_path / 'episode', 'hybrid', {
        'seed': 0, 'bottle_pose': None, 'max_steps': 2,
        'handoff_stage': 'before_left_arm',
    }, config)

    assert len(outgoing) == 1
    expected = format_mechanism_response([.2, .6])
    assert expected in outgoing[0]['prompt']
    run = json.loads((tmp_path / 'episode' / 'run.json').read_text())
    assert run['effective_prompt'] == outgoing[0]['prompt']


def test_setup_timeout_is_distinguished_from_policy_outcome(tmp_path, monkeypatch):
    from unittest.mock import patch
    config = load_settings()
    with patch('detent_bottle.gym_env.DetentBottleEnv._obs', return_value={}):
        final = run_episode(tmp_path / 'episode', 'hybrid', {
            'seed': 0, 'bottle_pose': None, 'max_steps': 1,
            'handoff_stage': 'before_left_arm',
        }, config)
    assert final['reason'] == 'time_limit'
    assert final['failure_category'] == 'setup_failure'
    assert not final['setup_complete']
    assert final['error'] is None
    trace = np.load(tmp_path / 'episode' / 'trajectory.npz')
    assert trace['eef_pos'].shape == (1, 2, 3)
    assert 'lid_deg' in trace and 'bottle_qpos' in trace
