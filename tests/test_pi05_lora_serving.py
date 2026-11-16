from dataclasses import dataclass

import pytest

from scripts.pi05.serve import configure_inference_model, validate_checkpoint_adapter_mode


@dataclass(frozen=True)
class Model:
    paligemma_variant: str = 'gemma_2b'
    action_expert_variant: str = 'gemma_300m'
    action_horizon: int = 15


@dataclass(frozen=True)
class Config:
    model: Model = Model()
    data: str = 'droid'


def test_lora_inference_selects_both_adapters_with_droid_transforms():
    original = Config()
    result = configure_inference_model(original, 'lora')
    assert result.model.paligemma_variant == 'gemma_2b_lora'
    assert result.model.action_expert_variant == 'gemma_300m_lora'
    assert result.data == original.data
    assert original.model.paligemma_variant == 'gemma_2b'
    assert configure_inference_model(original, 'full') is original


@pytest.mark.parametrize('mode,paths', [('full', ['llm/lora_a/value']), ('lora', ['llm/kernel/value'])])
def test_checkpoint_mode_mismatch_cannot_silently_drop_adapters(mode, paths):
    with pytest.raises(ValueError, match='disagree'):
        validate_checkpoint_adapter_mode(paths, mode)


def test_matching_checkpoint_modes_pass():
    assert validate_checkpoint_adapter_mode(['llm/kernel/value'], 'full') == 0
    assert validate_checkpoint_adapter_mode(['llm/kernel/value', 'llm/lora_a/value'], 'lora') == 1
