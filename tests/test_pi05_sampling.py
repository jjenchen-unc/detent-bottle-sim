import importlib.util
from pathlib import Path

import numpy as np
import pytest

from integration_pi05.sampling import (ControlledNoisePolicy, NOISE_KEY, NOISE_SHAPE,
                                       make_noise, noise_hash, sampling_metadata,
                                       validate_noise)


def test_noise_is_addressed_by_seed_and_request_not_global_rng():
    first = make_noise(17, 3)
    assert first.shape == (15, 32) and first.dtype == np.float32
    np.random.seed(100)
    np.random.normal(size=10000)
    np.testing.assert_array_equal(first, make_noise(17, 3))
    assert not np.array_equal(first, make_noise(17, 4))
    assert not np.array_equal(first, make_noise(18, 3))
    expected = np.random.Generator(np.random.PCG64(np.random.SeedSequence([17, 3]))).standard_normal(NOISE_SHAPE, dtype=np.float32)
    np.testing.assert_array_equal(first, expected)
    assert noise_hash(first) == noise_hash(np.asfortranarray(first))
    assert len(noise_hash(first)) == 64


@pytest.mark.parametrize('seed,index', [(True, 0), (0, False), (-1, 0), (0, -1), (2**32, 0), (0, 1.5)])
def test_noise_address_rejects_ambiguous_values(seed, index):
    with pytest.raises(ValueError):
        make_noise(seed, index)


@pytest.mark.parametrize('noise', [np.zeros((15, 8), np.float32), np.zeros((1, 15, 32), np.float32),
                                  np.zeros((15, 32), np.float64), np.full((15, 32), np.nan, np.float32)])
def test_noise_validation_rejects_wrong_dimensions_dtype_or_nonfinite(noise):
    with pytest.raises(ValueError, match='finite float32'):
        validate_noise(noise)


def test_wrapper_strips_transport_field_calls_public_keyword_and_preserves_request():
    calls = []
    class FakePolicy:
        metadata = {'configuration': 'pi05_droid'}
        def infer(self, observation, *, noise=None):
            calls.append((observation, noise))
            # Model owns this independent noise buffer.
            noise[0, 0] = 123
            return {'actions': np.zeros((15, 8))}
    wrapper = ControlledNoisePolicy(FakePolicy())
    noise = make_noise(0, 0)
    original = noise.copy()
    obs = {'observation/joint_position': np.zeros(7), NOISE_KEY: noise}
    assert wrapper.infer(obs)['actions'].shape == (15, 8)
    assert NOISE_KEY not in calls[0][0]
    assert calls[0][1][0, 0] == 123
    np.testing.assert_array_equal(obs[NOISE_KEY], original)
    assert wrapper.metadata['sampling'] == sampling_metadata()
    assert wrapper.metadata['configuration'] == 'pi05_droid'
    with pytest.raises(ValueError, match='requires'):
        wrapper.infer({'observation/joint_position': np.zeros(7)})
    assert len(calls) == 1


def test_wrapper_same_noise_remains_repeatable_across_interleaved_requests():
    class FakePolicy:
        def __init__(self):
            self.calls = 0
        def infer(self, obs, *, noise):
            self.calls += 1
            return {'actions': noise[:, :8] + obs['offset']}
    policy = FakePolicy()
    wrapper = ControlledNoisePolicy(policy)
    obs = {'offset': .25, NOISE_KEY: make_noise(5, 0)}
    first = wrapper.infer(obs)['actions']
    wrapper.infer({'offset': 1, NOISE_KEY: make_noise(99, 14)})
    repeated = wrapper.infer(obs)['actions']
    assert policy.calls == 3
    np.testing.assert_array_equal(first, repeated)


def test_sampling_module_can_be_loaded_as_standalone_staged_file():
    source = Path(__file__).resolve().parents[1] / 'integration_pi05' / 'sampling.py'
    spec = importlib.util.spec_from_file_location('standalone_sampling_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.sampling_metadata() == sampling_metadata()
    np.testing.assert_array_equal(module.make_noise(7, 2), make_noise(7, 2))


def test_standalone_server_loads_staged_module_without_project_install():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('standalone_server_test', root / 'scripts/pi05/serve.py')
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    sampling = server.load_sampling(root / 'integration_pi05/sampling.py')
    assert sampling.sampling_metadata() == sampling_metadata()
    with pytest.raises(ValueError, match='absolute staged'):
        server.load_sampling('sampling.py')


def test_server_identity_is_unique_and_records_gpu_runtime():
    import uuid
    from types import SimpleNamespace
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('server_identity_test',root/'scripts/pi05/serve.py')
    server = importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
    devices = [SimpleNamespace(id=0,platform='gpu',device_kind='NVIDIA test GPU')]
    first, second = server.server_identity(devices), server.server_identity(devices)
    assert uuid.UUID(first['server_instance_id']).version == 4
    assert first['server_instance_id'] != second['server_instance_id']
    assert first['gpu_devices'][0]['device_kind'] == 'NVIDIA test GPU'
    assert first['runtime_versions']['numpy'] == np.__version__
    assert first['hostname'] and first['server_pid'] > 0
