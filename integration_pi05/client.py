"""Bounded OpenPI RPCs, isolated from potentially blocking WebSocket code."""
import multiprocessing as mp
import datetime
import re
import time

import numpy as np


def _serve(requests, replies, host, port, policy_factory):
    """Only the disposable child imports OpenPI or owns a network connection."""
    events = []

    def emit(event):
        events.append(event)
        try:
            replies.send(('event', event, None))
        except (BrokenPipeError, EOFError, OSError):
            pass

    try:
        if policy_factory is None:
            from .websocket_policy import DiagnosedWebsocketClientPolicy
            policy = DiagnosedWebsocketClientPolicy(host=host, port=port, event_sink=emit)
        else:
            policy = policy_factory(host=host, port=port)
        metadata_getter = getattr(policy, 'get_server_metadata', None)
        metadata = metadata_getter() if callable(metadata_getter) else {}
        replies.send(('ready', metadata, None))
        while True:
            observation = requests.get()
            started = time.monotonic()
            emit({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'stage': 'inference_request', 'outcome': 'started'})
            try:
                result = policy.infer(observation)
            except Exception as exc:
                emit({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'stage': 'inference_request', 'outcome': 'failed',
                      'elapsed_s': time.monotonic() - started,
                      'error_type': type(exc).__name__, 'error': str(exc)})
                raise
            emit({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'stage': 'inference_request', 'outcome': 'succeeded',
                  'elapsed_s': time.monotonic() - started})
            replies.send(('result', result, None))
    except Exception as exc:
        try:
            replies.send(('error', f'{type(exc).__name__}: {exc}', None))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        replies.close()


CONTROLLED_POLICY = {
    'openpi_revision': '215abfb217dbac7d5f1273282331b9b1866c0479',
    'configuration': 'pi05_droid',
    'checkpoint': 'gs://openpi-assets/checkpoints/pi05_droid',
    'action_horizon': 15,
    'external_action_dim': 8,
}


def validate_expected_policy(contract=None):
    """Return a complete policy contract; candidate weights retain DROID semantics.

    An explicit contract replaces the whole stock contract, so omitted metadata
    cannot silently weaken provenance checks. This validates expectations only;
    it does not configure or select a checkpoint on the remote server.
    """
    contract = dict(CONTROLLED_POLICY) if contract is None else contract
    required = {'openpi_revision', 'configuration', 'checkpoint',
                'action_horizon', 'external_action_dim'}
    if not isinstance(contract, dict) or set(contract) != required:
        raise ValueError('expected_policy must contain exactly openpi_revision, configuration, checkpoint, action_horizon, external_action_dim')
    revision = contract['openpi_revision']
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('expected_policy.openpi_revision must be a full lowercase 40-character SHA')
    if contract['configuration'] != 'pi05_droid':
        raise ValueError('expected_policy.configuration must remain pi05_droid')
    checkpoint = contract['checkpoint']
    if not isinstance(checkpoint, str) or not checkpoint or checkpoint != checkpoint.strip():
        raise ValueError('expected_policy.checkpoint must be an exact nonempty checkpoint URI or path')
    for key, expected in (('action_horizon', 15), ('external_action_dim', 8)):
        if type(contract[key]) is not int or contract[key] != expected:
            raise ValueError(f'expected_policy.{key} must remain {expected}')
    return dict(contract)


class Client:
    """Synchronous client with a killable connection/inference worker.

    OpenPI is imported lazily in a spawned process. Connection setup and each
    inference have separate deadlines. After any RPC failure the client closes;
    construct a new instance to reconnect rather than reusing stale responses.
    ``policy_factory`` is a spawn-picklable testing seam, not a model dependency.
    """
    def __init__(self, host='localhost', port=8000, timeout=60,
                 connection_timeout=None, *, policy_factory=None):
        self.host, self.port = host, port
        self.timeout = self._positive(timeout)
        self.connection_timeout = self._positive(timeout if connection_timeout is None else connection_timeout)
        self.latencies = []
        self.server_metadata = {}
        self._factory = policy_factory
        self._process = self._requests = self._replies = None
        self._closed = False
        self._expected_instance = None
        self._expected_policy = validate_expected_policy()
        self._controlled = False
        self.transport_events = []

    @staticmethod
    def _positive(value):
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Timeout must be finite and positive')
        return value

    def _receive(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._replies.poll(remaining):
                self.transport_events.append({
                    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'stage': 'client_wait', 'outcome': 'timeout', 'timeout_s': timeout,
                })
                raise TimeoutError(f'OpenPI RPC exceeded {timeout:g} seconds')
            try:
                kind, payload, detail = self._replies.recv()
            except (EOFError, OSError) as exc:
                raise RuntimeError('OpenPI worker exited without a response') from exc
            if kind == 'event':
                self.transport_events.append(payload)
                continue
            if detail:
                self.transport_events.extend(detail)
            if kind == 'error':
                raise RuntimeError(f'OpenPI worker failed: {payload}')
            return kind, payload

    def _start(self):
        ctx = mp.get_context('spawn')
        self._requests = ctx.Queue(maxsize=1)
        self._replies, child_reply = ctx.Pipe(duplex=False)
        self._process = ctx.Process(target=_serve, args=(self._requests, child_reply,
                                   self.host, self.port, self._factory), daemon=True)
        try:
            self._process.start()
        finally:
            child_reply.close()
        kind, metadata = self._receive(self.connection_timeout)
        if kind != 'ready':
            raise RuntimeError('Invalid OpenPI worker handshake')
        if not isinstance(metadata, dict):
            raise ValueError('OpenPI server metadata must be a dictionary')
        self.server_metadata = metadata

    def connect(self, *, controlled=False, expected_instance=None, expected_policy=None):
        """Establish metadata handshake without transmitting an inference request."""
        if self._closed:
            raise RuntimeError('Client is closed; create a new client to reconnect')
        try:
            if expected_policy is not None:
                if not controlled:
                    raise ValueError('expected_policy requires a controlled connection')
                self._expected_policy = validate_expected_policy(expected_policy)
            if self._process is None:
                self._start()
            if expected_instance is not None:
                self._expected_instance = expected_instance
            if self._expected_instance is not None and self.server_metadata.get('server_instance_id') != self._expected_instance:
                raise ValueError('Connected server instance mismatch')
            if controlled:
                from .sampling import sampling_metadata
                expected = sampling_metadata()
                actual = self.server_metadata.get('sampling')
                if not isinstance(actual, dict) or any(actual.get(k) != v for k, v in expected.items()):
                    raise ValueError('Server does not advertise the required explicit-noise sampling protocol')
                mismatches = [key for key, value in self._expected_policy.items()
                              if key not in self.server_metadata or self.server_metadata[key] != value]
                if mismatches:
                    raise ValueError(f'Controlled policy provenance mismatch: {mismatches}')
                self._controlled = True
            return self.server_metadata
        except BaseException:
            self.close()
            raise

    def infer(self, observation):
        if self._closed:
            raise RuntimeError('Client is closed; create a new client to reconnect')
        start = time.monotonic()
        self.transport_events.append({
            'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'stage': 'inference_rpc', 'outcome': 'started',
        })
        try:
            if self._controlled or self._expected_instance is not None:
                self.connect(controlled=self._controlled, expected_instance=self._expected_instance)
            elif self._process is None:
                self._start()
            # Queue feeding uses a daemon thread, so a blocked child cannot block
            # the caller while it transmits the image payload.
            self._requests.put_nowait(observation)
            kind, result = self._receive(self.timeout)
            if kind != 'result':
                raise RuntimeError('Invalid OpenPI worker response')
            a = np.asarray(result['actions'])
            if a.shape != (15, 8) or not np.isfinite(a).all():
                raise ValueError('Checkpoint must return finite (15,8) actions')
        except BaseException:
            self.close()
            raise
        self.latencies.append(time.monotonic() - start)
        self.transport_events.append({
            'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'stage': 'inference_rpc', 'outcome': 'succeeded',
            'elapsed_s': time.monotonic() - start,
        })
        return a

    def close(self):
        self._closed = True
        if self._process is not None and self._process.pid is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=.5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=.5)
        if self._requests is not None:
            self._requests.cancel_join_thread()
            self._requests.close()
            self._requests = None
        if self._replies is not None:
            self._replies.close()
            self._replies = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
