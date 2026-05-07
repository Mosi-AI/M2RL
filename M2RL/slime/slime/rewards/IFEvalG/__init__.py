import logging
from collections.abc import Sequence
from typing import Any

from .evaluation_main import test_instruction_following_strict, InputExample

logger = logging.getLogger(__name__)
JsonDict = dict[str, Any]
KwargsDict = dict[str, str | int | float | None]

def _normalize_instruction_ids(raw_ids: Sequence[Any]) -> list[str]:
    """Ensure instruction identifiers are clean strings."""

    normalized: list[str] = []
    for entry in raw_ids or []:
        if entry is None:
            continue
        text = str(entry).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized

def _coerce_kwargs_list(
    raw_kwargs: Any,
    num_instructions: int,
) -> list[KwargsDict]:
    """Convert stored kwargs into the list structure expected by IFBench."""

    if isinstance(raw_kwargs, list):
        processed: list[KwargsDict] = []
        for entry in raw_kwargs:
            if isinstance(entry, dict):
                processed.append(dict(entry))
            else:
                processed.append({})
    elif isinstance(raw_kwargs, dict):
        processed = [dict(raw_kwargs) for _ in range(num_instructions)]
    else:
        processed = [{} for _ in range(num_instructions)]

    if len(processed) < num_instructions:
        tail = processed[-1] if processed else {}
        processed.extend([dict(tail) for _ in range(num_instructions - len(processed))])
    elif len(processed) > num_instructions:
        processed = processed[:num_instructions]

    # Remove explicit None values to match official preprocessing.
    sanitized: list[KwargsDict] = []
    for entry in processed:
        sanitized.append({k: v for k, v in entry.items() if v is not None})
    return sanitized

def _build_input_example(metadata: JsonDict) -> InputExample | None:
    instruction_ids = _normalize_instruction_ids(metadata.get("instruction_id_list"))
    if not instruction_ids:
        logger.debug("Missing instruction identifiers in metadata: %s", metadata)
        return None

    prompt_text = metadata.get("prompt_text")
    if prompt_text is None:
        prompt_text = ""
    else:
        prompt_text = str(prompt_text)

    raw_kwargs = metadata.get("kwargs")
    kwargs_list = _coerce_kwargs_list(raw_kwargs, len(instruction_ids))

    return InputExample(
        key=int(metadata.get("record_id") or 0),
        instruction_id_list=instruction_ids,
        prompt=prompt_text,
        kwargs=kwargs_list,
    )

def compute_ifevalg_reward(response: str, label: Any, metadata: JsonDict | None = None) -> float:
    """Score a model response using the official IFEvalG rules."""

    if metadata is None:
        logger.debug("No metadata provided for IFEvalG scoring.")
        return 0.0

    if response is None:
        return 0.0

    inp = _build_input_example(metadata)
    if inp is None:
        return 0.0

    output = test_instruction_following_strict(inp, response)
    return 1.0 if output.follow_all_instructions else 0.0