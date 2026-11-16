"""Standalone controlled-noise OpenPI server, executed inside a Slurm GPU job."""
import hashlib
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import time
import uuid
import importlib.metadata

import numpy as np


def configure_inference_model(train_config, finetune_mode):
    if finetune_mode == 'full':
        return train_config
    if finetune_mode != 'lora':
        raise ValueError('PI05_FINETUNE_MODE must be full or lora')
    return replace(train_config, model=replace(
        train_config.model, paligemma_variant='gemma_2b_lora',
        action_expert_variant='gemma_300m_lora'))


def validate_checkpoint_adapter_mode(parameter_paths, finetune_mode):
    if finetune_mode not in ('full', 'lora'):
        raise ValueError('PI05_FINETUNE_MODE must be full or lora')
    lora_count = sum('lora' in path.lower() for path in parameter_paths)
    if bool(lora_count) != (finetune_mode == 'lora'):
        raise ValueError('Checkpoint LoRA tensors and PI05_FINETUNE_MODE disagree; '
                         'refusing to load a model that could discard learned adapters')
    return lora_count


def load_sampling(path):
    path = Path(path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError('PI05_SAMPLING_MODULE must name an absolute staged sampling.py file')
    spec = importlib.util.spec_from_file_location('pi05_staged_sampling', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def server_identity(devices):
    versions = {'python': platform.python_version()}
    for name in ('numpy', 'jax', 'jaxlib'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'server_instance_id': str(uuid.uuid4()), 'server_pid': os.getpid(),
            'hostname': socket.gethostname(), 'runtime_versions': versions,
            'gpu_devices': [{'id': int(d.id), 'platform': d.platform,
                             'device_kind': str(d.device_kind), 'description': str(d)}
                            for d in devices if d.platform == 'gpu']}


def main():
    # Import expensive upstream dependencies only within the scheduled process.
    import jax
    import orbax.checkpoint as ocp
    from flax.traverse_util import flatten_dict
    from openpi.policies import policy_config
    from openpi.policies.droid_policy import make_droid_example
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config
    from openpi.shared import download

    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run this server inside an approved Slurm allocation')
    sampling_path = Path(os.environ['PI05_SAMPLING_MODULE'])
    sampling = load_sampling(sampling_path)
    run = Path(os.environ['PI05_RUN_DIR'])
    devices = jax.devices()
    if not any(d.platform == 'gpu' for d in devices):
        raise RuntimeError(f'No JAX GPU device: {devices}')
    identity = server_identity(devices)
    config_name = os.environ.get('PI05_CONFIG', 'pi05_droid')
    checkpoint = os.environ.get('PI05_CHECKPOINT', 'gs://openpi-assets/checkpoints/pi05_droid')
    finetune_mode = os.environ.get('PI05_FINETUNE_MODE', 'full')
    model_config = configure_inference_model(config.get_config(config_name), finetune_mode)
    if (model_config.model.action_horizon, model_config.model.action_dim) != sampling.NOISE_SHAPE:
        raise RuntimeError('Pinned model dimensions do not match the explicit-noise protocol')
    print('GPU devices:', devices, flush=True)
    checkpoint_dir = Path(download.maybe_download(checkpoint))
    with ocp.PyTreeCheckpointer() as checkpointer:
        parameter_paths = flatten_dict(checkpointer.metadata(checkpoint_dir / 'params')['params'], sep='/')
    lora_count = validate_checkpoint_adapter_mode(parameter_paths, finetune_mode)
    policy = sampling.ControlledNoisePolicy(policy_config.create_trained_policy(
        model_config, checkpoint_dir,
        sample_kwargs={'num_steps': sampling.NUM_STEPS}))
    np.random.seed(0)
    request = make_droid_example()
    noise = sampling.make_noise(0, 0)
    request[sampling.NOISE_KEY] = noise
    start = time.monotonic()
    actions = np.asarray(policy.infer(request)['actions'])
    if actions.shape != (15, 8) or not np.isfinite(actions).all():
        raise RuntimeError(f'Unexpected pi05_droid output: {actions.shape}')
    np.save(run / 'synthetic-actions.npy', actions)
    np.save(run / 'synthetic-noise.npy', noise)
    (run / 'warmup.json').write_text(json.dumps({
        'shape': list(actions.shape), 'finite': True,
        'seconds': time.monotonic() - start, 'noise_sha256': sampling.noise_hash(noise),
        **identity,
        'devices': [str(d) for d in devices], 'sampling': sampling.sampling_metadata(),
        'note': 'Synthetic request only; actual-observation sampling verification is separate.'}, indent=2))
    metadata = {**policy.metadata, **identity, 'openpi_revision': os.environ['OPENPI_REV'],
                'configuration': config_name,
                'checkpoint': checkpoint,
                'finetune_mode': finetune_mode,
                'model_variants': {'paligemma': model_config.model.paligemma_variant,
                                   'action_expert': model_config.model.action_expert_variant},
                'checkpoint_lora_tensors': lora_count,
                'action_horizon': 15, 'external_action_dim': 8,
                'rng': 'Explicit per-request noise through public Policy.infer(noise=); no private RNG mutation',
                'synthetic_warmup_calls': 1, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
                'sampling_module_sha256': hashlib.sha256(sampling_path.read_bytes()).hexdigest(),
                'server_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (run / 'server-metadata.json').write_text(json.dumps(metadata, indent=2))
    port = int(os.environ['PI05_PORT'])
    print(f'WARMUP PASSED; serving on 127.0.0.1:{port}', flush=True)
    WebsocketPolicyServer(policy, host='127.0.0.1', port=port, metadata=metadata).serve_forever()


if __name__ == '__main__':
    main()
