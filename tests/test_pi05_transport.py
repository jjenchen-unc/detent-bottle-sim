"""Actual OpenPI msgpack/WebSocket transport against a disposable local server."""
import multiprocessing as mp

import numpy as np
import pytest

pytest.importorskip('openpi_client')
pytest.importorskip('websockets.sync.server')

from integration_pi05.client import Client

METADATA = {'config': 'pi05_droid', 'action_horizon': 15, 'test_server': True}


def _dummy_server(ready):
    from openpi_client import msgpack_numpy
    from websockets.sync.server import serve

    def handle(connection):
        connection.send(msgpack_numpy.packb(METADATA))
        for message in connection:
            observation = msgpack_numpy.unpackb(message)
            image = observation['observation/exterior_image_1_left']
            joints = observation['observation/joint_position']
            assert image.shape == (224, 224, 3) and image.dtype == np.uint8
            assert image[0, 0, 0] == 123
            assert joints.shape == (7,)
            actions = np.zeros((15, 8), dtype=np.float32)
            actions[:, :7] = joints
            actions[:, 7] = observation['observation/gripper_position'][0]
            connection.send(msgpack_numpy.packb({'actions': actions}))

    try:
        with serve(handle, '127.0.0.1', 0, compression=None, max_size=None) as server:
            ready.send(('ready', server.socket.getsockname()[1]))
            ready.close()
            server.serve_forever()
    except Exception as exc:
        try:
            ready.send(('error', repr(exc)))
        except (OSError, EOFError):
            pass
        raise
    finally:
        ready.close()


@pytest.fixture
def websocket_server():
    context = mp.get_context('spawn')
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_dummy_server, args=(child,), daemon=True)
    process.start()
    child.close()
    try:
        assert parent.poll(10), 'Local test server failed to start'
        status, payload = parent.recv()
        assert status == 'ready', f'Local test server could not bind: {payload}'
        yield payload
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join(2)
        assert not process.is_alive(), 'Test server failed to terminate'
        process.close()


def test_actual_openpi_websocket_metadata_and_numpy_roundtrip(websocket_server):
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    image[0, 0, 0] = 123
    observation = {
        'observation/exterior_image_1_left': image,
        'observation/wrist_image_left': image.copy(),
        'observation/joint_position': np.linspace(-.3, .3, 7),
        'observation/gripper_position': np.array([.25]),
        'prompt': 'Open the bottle',
    }
    with Client(host='127.0.0.1', port=websocket_server,
                connection_timeout=10, timeout=5) as client:
        # No testing policy_factory: this imports the installed upstream client.
        first = client.infer(observation)
        assert client.server_metadata == METADATA
        assert first.shape == (15, 8) and first.dtype == np.float32
        np.testing.assert_allclose(first[:, :7], np.broadcast_to(observation['observation/joint_position'], (15, 7)), atol=1e-7)
        np.testing.assert_array_equal(first[:, 7], .25)
        process = client._process
        observation['observation/gripper_position'] = np.array([.75])
        second = client.infer(observation)
        np.testing.assert_array_equal(second[:, 7], .75)
        assert client._process is process
        assert len(client.latencies) == 2
    assert not process.is_alive()
    assert client.server_metadata == METADATA
