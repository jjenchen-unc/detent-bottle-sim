"""Exercise network failures in real spawned processes without OpenPI/GPU."""
import time

import numpy as np
import pytest

from integration_pi05.client import Client


class FakePolicy:
    def __init__(self, **kwargs):
        pass

    def infer(self, obs):
        if obs.get('hang'):
            time.sleep(30)
        if obs.get('error'):
            raise ValueError('server rejected observation')
        if obs.get('crash'):
            import os
            os._exit(2)
        return {'actions': np.full(obs.get('shape', (15, 8)), obs.get('value', 1.))}


class HangingConnection:
    def __init__(self, **kwargs):
        time.sleep(30)


class MetadataPolicy(FakePolicy):
    def get_server_metadata(self):
        return {'revision': 'pinned-test', 'action_horizon': 15,
                'sampling': {'rng_controlled': False}}


class InvalidMetadataPolicy(FakePolicy):
    def get_server_metadata(self):
        return ['invalid handshake metadata']


def test_success_reuses_worker_and_closes():
    client = Client(policy_factory=FakePolicy, timeout=5)
    assert client._process is None
    np.testing.assert_array_equal(client.infer({}), np.ones((15, 8)))
    assert client.server_metadata == {}
    pid = client._process.pid
    client.infer({'value': 2})
    assert client._process.pid == pid
    assert len(client.latencies) == 2
    client.close()
    client.close()
    assert not client._process.is_alive()
    with pytest.raises(RuntimeError, match='closed'):
        client.infer({})


@pytest.mark.parametrize('factory,obs,timeout', [
    (HangingConnection, {}, .2), (FakePolicy, {'hang': True}, .2),
])
def test_timeout_kills_worker(factory, obs, timeout):
    client = Client(policy_factory=factory, timeout=timeout,
                    connection_timeout=.5 if factory is HangingConnection else 5)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        client.infer(obs)
    assert time.monotonic() - start < 7
    assert not client._process.is_alive()
    assert client.latencies == []


@pytest.mark.parametrize('obs,error', [
    ({'shape': (10, 8)}, ValueError), ({'value': float('nan')}, ValueError),
    ({'error': True}, RuntimeError), ({'crash': True}, RuntimeError),
])
def test_bad_response_or_worker_failure_closes(obs, error):
    client = Client(policy_factory=FakePolicy, timeout=5)
    with pytest.raises(error):
        client.infer(obs)
    assert not client._process.is_alive()


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
def test_invalid_deadline(timeout):
    with pytest.raises(ValueError):
        Client(timeout=timeout)


def test_public_server_metadata_survives_worker_and_close():
    with Client(policy_factory=MetadataPolicy, timeout=5) as client:
        assert client.server_metadata == {}
        client.infer({})
        assert client.server_metadata == {'revision': 'pinned-test', 'action_horizon': 15,
                                          'sampling': {'rng_controlled': False}}
    assert client.server_metadata['revision'] == 'pinned-test'


def test_malformed_server_metadata_closes_client():
    client = Client(policy_factory=InvalidMetadataPolicy, timeout=5)
    with pytest.raises(ValueError, match='metadata must be a dictionary'):
        client.infer({})
    assert not client._process.is_alive()
    assert client.server_metadata == {}


class ControlledMetadataPolicy(FakePolicy):
    def get_server_metadata(self):
        from integration_pi05.sampling import sampling_metadata
        from integration_pi05.client import CONTROLLED_POLICY
        return {**CONTROLLED_POLICY, 'sampling':sampling_metadata()}


def test_controlled_connect_validates_without_inference():
    with Client(policy_factory=ControlledMetadataPolicy, timeout=5) as client:
        client.connect(controlled=True)
        assert client.latencies == []
        assert client._process.is_alive()


def test_controlled_connect_rejects_legacy_before_inference():
    client = Client(policy_factory=MetadataPolicy, timeout=5)
    with pytest.raises(ValueError, match='sampling protocol'):
        client.connect(controlled=True)
    assert client.latencies == []
    assert not client._process.is_alive()


class MismatchedControlledPolicy(ControlledMetadataPolicy):
    def get_server_metadata(self):
        metadata = super().get_server_metadata()
        metadata['checkpoint'] = 'wrong-checkpoint'
        return metadata


def test_controlled_connect_rejects_policy_provenance():
    client = Client(policy_factory=MismatchedControlledPolicy, timeout=5)
    with pytest.raises(ValueError, match='provenance mismatch'):
        client.connect(controlled=True)
    assert client.latencies == []
    assert not client._process.is_alive()


class InstanceControlledPolicy(ControlledMetadataPolicy):
    def get_server_metadata(self):
        return {**super().get_server_metadata(), 'server_instance_id':'persistent-A'}


def test_expected_instance_is_optional_but_fail_closed_when_supplied():
    with Client(policy_factory=InstanceControlledPolicy, timeout=5) as client:
        client.connect(controlled=True, expected_instance='persistent-A')
        assert client.infer({}).shape == (15,8)
    client = Client(policy_factory=InstanceControlledPolicy, timeout=5)
    with pytest.raises(ValueError,match='server instance mismatch'):
        client.connect(controlled=True,expected_instance='restarted-B')
    assert not client._process.is_alive() and client.latencies == []
