import asyncio
import copy
import json
import logging
import aiohttp
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import List, Dict, Any, Optional, Union, Callable

from slime.rollout.workplace_assistant.openai_tool_adapter import create_openai_adapter
# from tau_bench.agents.base import Agent
# from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME, ToolCallingAgent
from slime.rollout.workplace_assistant.types import Action #, RunConfig
from transformers import AutoTokenizer

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Set up logger for this module
logger = logging.getLogger(__name__)


def retry_on_error(max_retries: int = 3, return_default_on_failure: Any = None):
    """Decorator for retry logic with exponential backoff."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry_count = 0
            last_error = None
            
            while retry_count < max_retries:
                try:
                    return await func(*args, **kwargs)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_error = e
                    retry_count += 1
                    if retry_count >= max_retries:
                        break
                    # Exponential backoff
                    await asyncio.sleep(2 ** retry_count)
            
            # If all retries failed and return_default_on_failure is set, return it
            if return_default_on_failure is not None:
                logger.warning(f"All {max_retries} retries failed for {func.__name__}: {last_error}")
                return return_default_on_failure
            raise last_error
        return wrapper
    return decorator


class Status(Enum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


@dataclass
class InteractionResult:
    prompt: str
    reward: float
    messages: list[dict[str, Any]]
    info: dict[str, Any]
    response: str = ""
    response_log_probs: list[float] | None = None
    loss_mask: list[int] | None = None
    tokens: int | None = None
    status: Status = Status.COMPLETED


def call_to_action_sglang(calls: list[Any], text_response: str) -> Action:
    """
    Convert sglang response message to Action, similar to original message_to_action
    but adapted for sglang response format.
    """
    actions = []
    if calls:
        for tool_call in calls:
            params = json.loads(tool_call["parameters"])
            if not isinstance(params, dict):
                logger.warning(f"{params} does not follow dict structure for action")
            else:
                action = Action(name=tool_call["name"], type='function_call', arguments=params)
                actions.append(action)
    else:
        # Default action if no action was found.
        actions = [Action(name='respond', type='text', arguments={"content": text_response})]
    return actions


TOOL_INSTRUCTION = (
    " At each turn, you are allowed to call one or no function to assist "
    "with task execution using <tools></tools> XML tags.\n"
    "YOU MUST EXECUTE TOOLS TO MAKE ANY MODIFICATIONS OR CANCELLATIONS. "
    "Each tool call leads to a message returned by the system.\n"
    "NEVER confirm execution to the user without seeing confirmation "
    "from the tool system.\n"
)


class TrainableAgentMixin:
    """
    Mixin class that provides trainable agent functionality for tau-bench environments.

    This mixin extends the original tau-bench agent with async LLM interaction
    capabilities for reinforcement learning training using sglang servers.
    """

    def _reformulate_tool_call(self, text: str) -> str:
        """
        Reformulate tool call instruction for tau-bench environment.

        The default tool template assumes one or more function calls, but for
        tau-bench, at most one tool call or skip tool calls are the valid options.

        Args:
            text: Original tool instruction text

        Returns:
            Reformulated tool instruction text
        """
        return text.replace("You may call one or more functions to assist with the user query.", TOOL_INSTRUCTION)

    async def _call_llm(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Make an LLM call tracking.

        Args:
            url: SGLang server URL
            payload: Request payload containing text and sampling parameters

        Returns:
            LLM response from sglang server
        """
        return await post(url, payload)

    def _parse_tool(self, response: str) -> dict[str, Any]:
        """
        Parse tool calls from LLM response string.

        Args:
            response: Raw response text from sglang

        Returns:
            Parsed tool call result in OpenAI format
        """
        response = response.split("</think>")[-1].strip()  # Extract the part after </think> if present
        return self.openai_adapter.parse_response_to_openai_format(response)

    def _validate_tool_action(self, action: Action) -> Optional[str]:
        """
        Validate tool name and parameters against tools_info definition.

        Args:
            action: Action to validate

        Returns:
            None if validation passes, error message string if validation fails
        """
        # Validate tool name
        tool_found = None
        for tool in self.tools_info:
            if tool.get("type") == "function" and tool.get("function", {}).get("name") == action.name:
                tool_found = tool.get("function", {})
                break
            elif tool.get("name") == action.name:
                tool_found = tool
                break
        
        if tool_found is None:
            # Tool name not found in tools_info
            return f"Model attempted to call undefined function: {action.name}"
        
        # Validate parameters against schema if available
        if "parameters" in tool_found and tool_found["parameters"] is not None:
            schema = tool_found["parameters"]
            if isinstance(schema, dict) and "properties" in schema:
                required_params = schema.get("required", [])
                valid_params = set(schema.get("properties", {}).keys())
                
                # Check for unexpected parameters
                for param_name in action.arguments.keys():
                    if param_name not in valid_params:
                        return f"{action.name}() got an unexpected keyword argument '{param_name}'"
                
                # Check for missing required parameters
                for required_param in required_params:
                    if required_param not in action.arguments:
                        return f"{action.name}() missing required argument '{required_param}'"
        
        return None

    @retry_on_error(max_retries=5)
    async def _execute_tool(self, session, action: Action):
        """Execute a tool/action in the environment."""
        url = f"{self.resource_server_url}/{action.name}"
        payload = {**action.arguments}
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            output = await resp.json()
            return output

    @retry_on_error(max_retries=5)
    async def initialize_session(self, session):
        """Initialize the session."""
        url = self.resource_server_url + "/seed_session"
        async with session.post(url, json={}, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            resp.raise_for_status()

    def _build_initial_messages(self, prompt) -> list[dict[str, Any]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            return prompt
        else:
            raise ValueError("Prompt must be a string or list of messages.")

    def _prepare_prompt_tokens(self, state: GenerateState, messages: list[dict[str, Any]]) -> tuple[str, list[int]]:
        """
        Prepare prompt text and tokenize it.

        Args:
            state: GenerateState instance with tokenizer
            messages: Conversation messages

        Returns:
            Tuple of (prompt_text, prompt_token_ids)
        """
        prompt_text = state.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=self.tools_info
        )
        # Reformulate tool call instruction for tau-bench
        prompt_text = self._reformulate_tool_call(prompt_text)
        prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        return prompt_text, prompt_token_ids

    async def asolve(
        self,
        rollout_args: dict[str, Any],
        sample: Sample,
        sampling_params: dict[str, Any],
        max_num_steps: int = 6,
    ) -> InteractionResult:
        """
        Execute async agent-environment interaction for training.

        This method extends the original Agent to support async interaction with LLM
        server for reinforcement learning training. It maintains conversation history,
        tracks tokens, and records metadata for training purposes.

        Args:
            env: Tau-bench environment instance
            rollout_args: Rollout configuration arguments
            sampling_params: LLM sampling parameters
            task_index: Specific task index to solve (optional)
            max_num_steps: Maximum number of interaction steps

        Returns:
            InteractionResult containing the complete interaction trajectory
        """
        # Initialize environment and state
        state = GenerateState(rollout_args)
        url = f"http://{rollout_args.sglang_router_ip}:" f"{rollout_args.sglang_router_port}/generate"

        # Build initial conversation
        messages = self._build_initial_messages(sample.prompt)
        prompt_text, prompt_token_ids = self._prepare_prompt_tokens(state, messages)

        # Initialize tracking variables
        loss_masks = []
        response_token_ids = []
        response_log_probs = []
        total_reward = 0.0
        action_seq = []
        sampling_params = copy.deepcopy(sampling_params)

        # Initialize result
        res = InteractionResult(prompt=prompt_text, reward=0, messages=[], info={})

        # Multi-turn interaction loop
        async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as session:
            await self.initialize_session(session)
            for _ in range(max_num_steps):
                # Prepare payload for sglang
                text_input = state.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, tools=self.tools_info
                )
                # Reformulate tool call instruction for tau-bench
                text_input = self._reformulate_tool_call(text_input)
                payload = {
                    "text": text_input,
                    "sampling_params": sampling_params,
                    "return_logprob": True
                }

                # Send request to sglang server
                output = await self._call_llm(url, payload)

                # Check for abort
                if output["meta_info"]["finish_reason"]["type"] == "abort":
                    res.status = Status.ABORTED
                    return self._build_final_result(
                        res, total_reward, messages, loss_masks, prompt_token_ids, response_token_ids, response_log_probs
                    )

                response = output["text"]
                # Remove end of conversation token if present
                if response.endswith("<|im_end|>"):
                    response = response[:-10]

                # Parse tool calls using OpenAI adapter
                logger.debug(f"Using OpenAI adapter to parse response: {response[:100]}...")
                try:
                    openai_result = self._parse_tool(response)
                    logger.debug(f"OpenAI adapter result: success={openai_result['success']}")

                    if not openai_result["success"]:
                        logger.warning(f"OpenAI adapter failed: {openai_result['error']}")
                        logger.warning(
                            f"rollout response: {response} can not be parsed into " f"tool calls {openai_result['error']}"
                        )
                        res.status = Status.ABORTED
                        return self._build_final_result(
                            res, total_reward, messages, loss_masks, prompt_token_ids, response_token_ids, response_log_probs
                        )

                    # Extract parsed results
                    parsed = openai_result["parsed_result"]
                    logger.debug(
                        f"Successfully parsed - normal_text: '{parsed['normal_text']}', " f"calls: {parsed['calls']}"
                    )

                except Exception as e:
                    logger.warning(f"Exception in OpenAI adapter: {e}")
                    logger.warning(f"rollout response: {response} can not be parsed into " f"tool calls {e}")
                    res.status = Status.ABORTED
                    return self._build_final_result(
                        res, total_reward, messages, loss_masks, prompt_token_ids, response_token_ids, response_log_probs
                    )
                # print('#' * 50)
                # print('######## Prompt\n', text_input)
                # print('######## Response\n', response)
                # print('######## Messages\n', '\n'.join([str(m) for m in messages]))
                # print('######## Parsed\n', openai_result)

                # Add assistant response to conversation
                messages.append({"role": "assistant", "content": response})
                # assistant_token_ids, assistant_loss_mask, _ = self._get_token_delta(state.tokenizer, messages, 1)
                assistant_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
                assistant_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
                assistant_loss_mask = [1] * len(assistant_token_ids)
                response_token_ids.extend(assistant_token_ids)
                response_log_probs.extend(assistant_log_probs)
                loss_masks.extend(assistant_loss_mask)
                sampling_params["max_new_tokens"] -= len(assistant_token_ids)

                if len(parsed["calls"]) == 0:
                    logger.debug("No tool call detected, finishing interaction.")
                    res.status = Status.COMPLETED
                    break

                # Execute action in environment
                agent_content, calls = parsed["normal_text"], parsed["calls"]
                logger.debug(f"Creating action from - content: '{agent_content}', " f"calls: {calls}")
                actions = call_to_action_sglang(calls, agent_content)
                if len(actions) == 0:
                    logger.warning(f"No valid actions parsed from calls: {repr(calls)}")
                    res.status = Status.ABORTED
                    return self._build_final_result(
                        res, total_reward, messages, loss_masks, prompt_token_ids, response_token_ids, response_log_probs
                    )

                logger.debug(f"Created actions: {actions}")
                for action in actions:
                    # Validate tool name and parameters before executing
                    validation_error = self._validate_tool_action(action)
                    if validation_error is not None:
                        # Validation failed, set env_response to error message
                        logger.warning(f"Tool validation failed: {validation_error}")
                        env_response = validation_error
                    else:
                        # Validation passed, execute the tool
                        try:
                            env_response = await self._execute_tool(session, action)
                            env_response = env_response["output"] if isinstance(env_response, dict) and "output" in env_response else env_response
                            env_response = json.dumps(env_response) if isinstance(env_response, (dict, list)) else str(env_response)
                            if env_response.startswith('Error executing tool'):
                                print(action)
                        except Exception as e:
                            logger.warning("Environment step failed, this is usually related to " "the User simulation call.")
                            logger.warning(f"Error: {e}")
                            res.status = Status.ABORTED
                            return self._build_final_result(
                                res, total_reward, messages, loss_masks, prompt_token_ids, response_token_ids, response_log_probs
                            )

                    # print('######## Env Response\n', env_response)

                    # Update message history based on action type
                    action_seq.append(action)
                    if action.name == 'respond':
                        # Direct response from user
                        messages.append({"role": "user", "content": env_response})
                    else:
                        messages.append(
                            {
                                "role": "tool",
                                "name": action.name,
                                "content": env_response,
                            }
                        )

                # Update token tracking
                env_token_ids, env_loss_mask, env_log_probs = self._get_token_delta(state.tokenizer, messages, len(actions))
                if len(env_token_ids) >= sampling_params["max_new_tokens"]:
                    res.status = Status.TRUNCATED
                    break
                else:
                    sampling_params["max_new_tokens"] -= len(env_token_ids)
                response_token_ids.extend(env_token_ids)
                loss_masks.extend(env_loss_mask)
                response_log_probs.extend(env_log_probs)
            else:
                res.status = Status.TRUNCATED

            # Call verify endpoint with retry logic
            payload = {
                "output": [action.model_dump() for action in action_seq],
                "ground_truth": sample.metadata["ground_truth_tool_calls"],
            }
            total_reward = await self._call_verify(session, payload)

            # from pprint import pprint
            # pprint(messages)
            # print('reward:', total_reward)
            return self._build_final_result(
                res, total_reward, messages, loss_masks, prompt_token_ids, response_token_ids, response_log_probs
            )

    def _get_token_delta(self, tokenizer: AutoTokenizer, messages: list[dict], num_new_message: int) -> tuple[list[int], list[int]]:
        """
        Calculate token delta for multi-turn conversations.

        Tokenization logic adapted from:
        https://verl.readthedocs.io/en/v0.4.1/sglang_multiturn/multiturn.html
        to calculate the right token count in a multi-turn environment using
        delta between messages.

        Args:
            tokenizer: Tokenizer instance
            messages: Conversation messages

        Returns:
            Tuple of (token_ids, loss_mask)
        """
        curr = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        token_ids = []
        loss_mask = []

        # Case 1: last message is an assistant response
        if messages[-1]["role"] == "assistant":
            prev = tokenizer.apply_chat_template(messages[:-num_new_message], add_generation_prompt=True, tokenize=False)
            new_tokens = tokenizer.encode(curr[len(prev) :], add_special_tokens=False)
            token_ids += new_tokens
            loss_mask += [1] * len(new_tokens)  # Mask only the new assistant tokens
        else:
            # Case 2: last message is a tool response or environment observation
            prev = tokenizer.apply_chat_template(messages[:-num_new_message], add_generation_prompt=False, tokenize=False)
            new_tokens = tokenizer.encode(curr[len(prev) :], add_special_tokens=False)
            token_ids += new_tokens
            loss_mask += [0] * len(new_tokens)  # Don't mask environment/tool tokens
        
        log_probs = [0.0] * len(token_ids)  # Placeholder for log probabilities

        return token_ids, loss_mask, log_probs

    @retry_on_error(max_retries=5, return_default_on_failure=0.0)
    async def _call_verify(self, session, payload):
        """Call verify endpoint with retry logic, return 0.0 on failure."""
        url = f"{self.resource_server_url}/verify"
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            resp.raise_for_status()
            result = await resp.json()
            print(result)
            return result['reward']

    def _build_final_result(
        self,
        res: InteractionResult,
        total_reward: float,
        messages: list[dict[str, Any]],
        loss_masks: list[int],
        prompt_token_ids: list[int],
        response_token_ids: list[int],
        response_log_probs: list[float],
    ) -> InteractionResult:
        """
        Build the final interaction result with all collected data.

        Args:
            res: InteractionResult instance to populate
            total_reward: Total reward accumulated during interaction
            info: Environment info dictionary
            messages: Complete conversation messages
            loss_masks: Loss masks for training
            prompt_token_ids: Prompt token IDs
            response_token_ids: Response token IDs

        Returns:
            Populated InteractionResult
        """
        res.reward = total_reward
        res.messages = messages
        res.loss_mask = loss_masks
        res.tokens = prompt_token_ids + response_token_ids
        res.response = "".join([msg.get("content", "") for msg in messages if msg["role"] == "assistant"])
        res.response_log_probs = response_log_probs
        res.response_length = len(loss_masks)

        logger.debug(
            f"_build_final_result: response_length={res.response_length}, "
            f"response_loss_mask_len={len(loss_masks)}, "
            f"prompt_token_len={len(prompt_token_ids)}, "
            f"response_token_len={len(response_token_ids)}, "
            f"response='{res.response[:100]}...'"
        )
        return res


class TrainableToolCallingAgent(TrainableAgentMixin):
    """
    A trainable version of ToolCallingAgent that uses sglang rollout for training.

    This agent combines the original ToolCallingAgent functionality with the
    TrainableAgentMixin to support async interaction with sglang servers for
    reinforcement learning training.
    """

    def __init__(
        self,
        tools_info: list[dict[str, Any]],
        resource_server_url: str,
        temperature: float = 1.0,
        rollout_args: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ):
        self.tools_info = tools_info
        self.resource_server_url = resource_server_url
        # Store rollout and sampling parameters as instance variables
        self.rollout_args = rollout_args or {
            "sglang_router_ip": "127.0.0.1",
            "sglang_router_port": 30000,
            "use_http2": False,
        }
        self.sampling_params = sampling_params or {
            "temperature": temperature,
            "max_new_tokens": 4096,
            "top_p": 0.95,
        }
        # Initialize OpenAI adapter
        self.openai_adapter = create_openai_adapter(tools_info=self.tools_info, parser_type="qwen25")


def agent_factory(
    tools_info: list[dict[str, Any]],
    config,
    rollout_args: dict[str, Any] | None = None,
    sampling_params: dict[str, Any] | None = None,
):
    if config.agent_strategy == "tool-calling":
        return TrainableToolCallingAgent(
            tools_info=tools_info,
            resource_server_url=config.resources_server_url,
            temperature=config.temperature,
            rollout_args=rollout_args,
            sampling_params=sampling_params,
        )
    else:
        raise NotImplementedError(f"Unsupported agent strategy: {config.agent_strategy}")
