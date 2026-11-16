import json
import numpy as np
import pytest
from integration_pi05.preflight import run_preflight, sampled_request
from integration_pi05.sampling import NOISE_KEY, sampling_metadata, noise_hash


class FakeClient:
    latencies = []
    from integration_pi05.client import CONTROLLED_POLICY
    server_metadata = {**CONTROLLED_POLICY, 'sampling':sampling_metadata()}
    def __init__(self, folder, drift=False): self.folder=folder; self.drift=drift; self.calls=0
    def connect(self, **kwargs): assert kwargs['controlled']
    def infer(self, payload):
        name = ('A-first', 'B', 'A-repeat')[self.calls]
        saved = np.load(self.folder / (name+'.npz'))
        np.testing.assert_array_equal(saved[NOISE_KEY], payload[NOISE_KEY])
        provenance = json.loads((self.folder / (name+'.json')).read_text())
        assert provenance['noise_sha256'] == noise_hash(payload[NOISE_KEY])
        value = float(payload[NOISE_KEY][0,0])
        if self.drift and self.calls == 2:
            value += .01
        self.calls += 1
        return np.full((15,8), value)


@pytest.mark.parametrize('drift', [False,True])
def test_preflight_saves_exact_requests_and_checks_a_agreement(tmp_path, drift):
    out = tmp_path / 'preflight'; client = FakeClient(out,drift)
    if drift:
        with pytest.raises(RuntimeError, match='preflight failed'):
            run_preflight(client, {'prompt':'test'}, out, 7)
    else:
        assert run_preflight(client, {'prompt':'test'}, out, 7)['passed']
    result = json.loads((out / 'result.json').read_text())
    assert result['passed'] != drift
    assert result['a_exact_equal'] != drift
    assert result['b_vs_a_max_abs_difference'] > 0
    assert len(list(out.glob('*-actions.npy'))) == 3


def test_request_noise_is_separate_and_repeatable():
    a, metadata = sampled_request({'prompt':'test'}, 2, 3)
    b, _ = sampled_request({'prompt':'test'}, 2, 3)
    np.testing.assert_array_equal(a[NOISE_KEY], b[NOISE_KEY])
    assert metadata['request_index'] == 3


@pytest.mark.parametrize('override', [None, 17])
def test_cli_uses_config_sampling_seed_and_frontview_without_required_flag(tmp_path, monkeypatch, override):
    from pathlib import Path
    from types import SimpleNamespace
    from integration_pi05 import preflight
    from integration_pi05.sampling import make_noise
    folder = tmp_path / 'preflight'
    closed = []
    class Env:
        def __init__(self, **kwargs):
            assert kwargs['cameras'] == ['frontview', 'left_wrist']
            self.sim = SimpleNamespace(steps=0, data=SimpleNamespace(qpos=np.zeros(3)))
        def reset(self, seed):
            return {'frontview': np.full((224,224,3), 55, np.uint8),
                    'left_wrist': np.full((224,224,3), 99, np.uint8),
                    'joint_pos': np.zeros(14), 'gripper_qpos': np.full(4,.045)}, {}
        def close(self): closed.append('env')
    class Client(FakeClient):
        def __init__(self, *args, **kwargs): super().__init__(folder)
        def infer(self, payload):
            assert payload['observation/exterior_image_1_left'][0,0,0] == 55
            assert payload['observation/wrist_image_left'][0,0,0] == 99
            return super().infer(payload)
        def close(self): closed.append('client')
    config = Path(__file__).resolve().parents[1] / 'configs/pi05/frontview.json'
    argv = ['preflight', '--config', str(config), '--out', str(folder)]
    if override is not None: argv += ['--sampling-seed', str(override)]
    monkeypatch.setattr('sys.argv', argv)
    monkeypatch.setattr('detent_bottle.gym_env.DetentBottleEnv', Env)
    monkeypatch.setattr('integration_pi05.client.Client', Client)
    preflight.main()
    expected_seed = 0 if override is None else override
    assert json.loads((folder / 'result.json').read_text())['sampling_seed'] == expected_seed
    with np.load(folder / 'A-first.npz') as data:
        np.testing.assert_array_equal(data[NOISE_KEY], make_noise(expected_seed,0))
    assert closed == ['client','env']
