"""
Workplace Assistant Integration for slime Training

This module provides the main interface for training agents in workplace assistant environments
using the slime framework. It handles agent-environment interactions and converts
results to the format expected by slime's training pipeline.
"""

import logging
import os
from typing import Any

from pydantic import BaseModel

from slime.rollout.workplace_assistant.trainable_agents import InteractionResult, Status, agent_factory
from slime.rollout.workplace_assistant.tool_info import WORKBENCH_TOOL_INFO
from slime.utils.types import Sample

# Set up logger for this module
logger = logging.getLogger(__name__)

# Workplace assistant configuration (will be overridden by --workplace-assistant-resources-server-url)
WORKPLACE_ASSISTANT_CONFIGS = {
    "resources_server_url": "http://localhost:12000"
}
# Replace with your actual API key for user sim

class RunConfig(BaseModel):
    resources_server_url: str
    temperature: float = 1.0
    agent_strategy: str = "tool-calling"

workplace_assistant_config = RunConfig(**WORKPLACE_ASSISTANT_CONFIGS)


def res_to_sample(res: InteractionResult, index: int) -> Sample:
    """
    Convert InteractionResult to Sample format for slime training.

    This function transforms the workplace assistant interaction result into the format
    expected by slime's training pipeline, handling status mapping and response
    length calculation.

    Args:
        res: InteractionResult from workplace assistant agent
        task_index: Index of the task being processed

    Returns:
        Sample object for slime training
    """
    # Map workplace assistant status to slime status
    status_mapping = {
        Status.COMPLETED: "completed",
        Status.TRUNCATED: "truncated",
        Status.ABORTED: "aborted",
    }
    status = status_mapping.get(res.status)

    # Debug logging for response tracking
    logger.debug(
        f"res_to_sample: response_length="
        f"{res.response_length if hasattr(res, 'response_length') else 'None'}, "
        f"loss_mask_len={len(res.loss_mask) if res.loss_mask else 'None'}, "
        f"tokens_len={len(res.tokens) if res.tokens else 'None'}"
    )

    # Create sample with basic information
    sample = Sample(
        index=index,
        prompt=res.prompt,
        tokens=res.tokens,
        response=res.response,
        rollout_log_probs=res.response_log_probs,
        reward=res.reward,
        loss_mask=res.loss_mask,
        status=status,
        metadata=res.info,
    )

    # Ensure response_length is set correctly
    if hasattr(res, "response_length"):
        sample.response_length = res.response_length
    else:
        # Fallback: calculate from loss_mask if available
        if res.loss_mask:
            # loss_mask only contains response part, so length equals response_length
            sample.response_length = len(res.loss_mask)
        elif res.tokens:
            # If no loss_mask available, use total tokens as fallback
            sample.response_length = len(res.tokens)
        else:
            sample.response_length = 0
            logger.debug(f"res_to_sample: Set response_length={sample.response_length}")

    return sample


async def generate(args: dict[str, Any], sample: Sample, sampling_params: dict) -> Sample:
    """
    Generate a complete agent-environment interaction trajectory for workplace assistant.

    This is the main entry point for slime training. It creates a workplace assistant
    environment, initializes a trainable agent, and executes a full interaction
    trajectory. The result is converted to slime's Sample format for training.

    Args:
        args: Rollout arguments from slime training pipeline
        sample: Sample containing task index in prompt field
        sampling_params: LLM sampling parameters

    Returns:
        Sample object containing the complete interaction trajectory

    Raises:
        AssertionError: If partial rollout is requested (not supported)
    """
    # Validate arguments
    assert not args.partial_rollout, "Partial rollout is not supported for workplace assistant interactions."

    # Update workplace assistant config with command-line argument if provided
    if hasattr(args, 'workplace_assistant_resources_server_url') and args.workplace_assistant_resources_server_url is not None:
        workplace_assistant_config.resources_server_url = args.workplace_assistant_resources_server_url

    # Create trainable agent
    workplace_assistant_config.temperature = args.rollout_temperature
    agent = agent_factory(
        config=workplace_assistant_config,
        tools_info=WORKBENCH_TOOL_INFO["tools"],
        rollout_args=args,
        sampling_params=sampling_params,
    )

    # Execute agent-environment interaction
    # Note: The sample.prompt field contains the task index for repeatability
    interaction_result = await agent.asolve(agent.rollout_args, sample, agent.sampling_params)

    # Convert to slime Sample format
    result_sample = res_to_sample(interaction_result, sample.index)

    return result_sample
