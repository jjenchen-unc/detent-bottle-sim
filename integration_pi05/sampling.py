"""Explicit inference noise protocol, independent of OpenPI and model imports.

This file can be staged standalone beside the Longleaf server script.
"""
import hashlib
from numbers import Integral

import numpy as np

PROTOCOL = 'pi05-explicit-noise-v1'
NOISE_KEY = 'sampling/noise'
NOISE_SHAPE = (15, 32)
NUM_STEPS = 10


def _uint32(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 2**32:
        raise ValueError(f'{name} must be an integer in [0, 2**32)')
    return int(value)


def make_noise(seed, request_index):
    """Generate independently addressable standard normal float32 noise."""
    sequence = np.random.SeedSequence([_uint32(seed, 'sampling_seed'),
                                      _uint32(request_index, 'request_index')])
    rng = np.random.Generator(np.random.PCG64(sequence))
    return rng.standard_normal(NOISE_SHAPE, dtype=np.float32)


def validate_noise(noise):
    noise = np.asarray(noise)
    if noise.shape != NOISE_SHAPE or noise.dtype != np.dtype('float32') or not np.isfinite(noise).all():
        raise ValueError('Explicit inference noise must be finite float32 (15,32)')
    return noise


def noise_hash(noise):
    """Hash exact canonical little-endian C-order float32 bytes."""
    noise = validate_noise(noise)
    return hashlib.sha256(noise.astype('<f4', copy=False).tobytes(order='C')).hexdigest()


def sampling_metadata():
    return {'protocol': PROTOCOL, 'noise_shape': list(NOISE_SHAPE),
            'noise_dtype': 'float32',
            'generator': 'numpy.PCG64/SeedSequence([sampling_seed,request_index])',
            'num_steps': NUM_STEPS}


class ControlledNoisePolicy:
    """Translate a WebSocket observation field to the public infer(noise=) API."""
    def __init__(self, policy):
        self.policy = policy

    @property
    def metadata(self):
        return {**(getattr(self.policy, 'metadata', None) or {}),
                'sampling': sampling_metadata()}

    def infer(self, observation):
        if NOISE_KEY not in observation:
            raise ValueError(f'Controlled sampling requires {NOISE_KEY}')
        noise = validate_noise(observation[NOISE_KEY]).copy()
        clean = {key: value for key, value in observation.items() if key != NOISE_KEY}
        return self.policy.infer(clean, noise=noise)
