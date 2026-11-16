"""Instrumented OpenPI WebSocket client with bounded pre-inference retries."""
import datetime
import time

import websockets.sync.client

from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy


class TransportHandshakeError(RuntimeError):
    """A fresh WebSocket/metadata handshake failed after bounded attempts."""


class DiagnosedWebsocketClientPolicy(WebsocketClientPolicy):
    """Keep OpenPI's message protocol while exposing each handshake stage."""

    def __init__(self, host="localhost", port=None, api_key=None, *, event_sink=None,
                 open_timeout=30, metadata_timeout=30, attempts=3):
        self.transport_events = []
        self._event_sink = event_sink
        self._open_timeout = float(open_timeout)
        self._metadata_timeout = float(metadata_timeout)
        self._handshake_attempts = int(attempts)
        if self._handshake_attempts < 1:
            raise ValueError("attempts must be positive")
        super().__init__(host=host, port=port, api_key=api_key)

    def _event(self, stage, outcome, attempt, started, **values):
        event = {
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "stage": stage,
            "outcome": outcome,
            "attempt": attempt,
            "elapsed_s": time.monotonic() - started,
            **values,
        }
        self.transport_events.append(event)
        if self._event_sink is not None:
            self._event_sink(event)

    def _wait_for_server(self):
        last_error = None
        events_start = len(self.transport_events)
        for attempt in range(1, self._handshake_attempts + 1):
            connection = None
            stage = "websocket_upgrade"
            started = time.monotonic()
            self._event(stage, "started", attempt, started)
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                connection = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=self._open_timeout,
                    proxy=None,
                )
                self._event(stage, "succeeded", attempt, started)

                stage = "server_metadata_read"
                started = time.monotonic()
                self._event(stage, "started", attempt, started)
                packed_metadata = connection.recv(timeout=self._metadata_timeout)
                metadata = msgpack_numpy.unpackb(packed_metadata)
                self._event(stage, "succeeded", attempt, started)
                return connection, metadata
            except Exception as exc:
                last_error = exc
                self._event(stage, "failed", attempt, started,
                            error_type=type(exc).__name__, error=str(exc))
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass
                if attempt < self._handshake_attempts:
                    time.sleep(0.5 * attempt)
        summary = self.transport_events[events_start:]
        raise TransportHandshakeError(
            f"WebSocket handshake failed after {self._handshake_attempts} attempts; "
            f"last stage={summary[-1]['stage'] if summary else 'unknown'}: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error
