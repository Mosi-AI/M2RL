from typing import Any

from slime.utils.types import Sample
from slime.rollout.workplace_assistant.generate_with_workplace_assistant import generate as generate_with_workplace_assistant
from slime.rollout.sglang_rollout import generate as sgl_generate

async def generate(args: dict[str, Any], sample: Sample, sampling_params: dict) -> Sample:
    if sample.metadata["rm_type"] == 'workbench':
        return await generate_with_workplace_assistant(args, sample, sampling_params)
    else:
        return await sgl_generate(args, sample, sampling_params)
