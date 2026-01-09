"""Integration for QwQ and most Qwen series chat models"""

import json
from json import JSONDecodeError
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Type,
)

import json_repair
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from pydantic import (
    Field,
)

from .base import _BaseChatQwen, _DictOrPydantic, _DictOrPydanticClass


class ChatQwQ(_BaseChatQwen):
    """Qwen QwQ Thinking chat model integration to access models hosted in Qwen QwQ Thinking's API.

    Setup:
        Install ``langchain-qwq`` and set environment variable ``DASHSCOPE_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-qwq
            export DASHSCOPE_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of Qwen QwQ Thinking model to use, e.g. "qwen-qwen2.5-coder-32b-instruct".
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.

    Key init args — client params:
        timeout: Optional[float]
            Timeout for requests.
        max_retries: int
            Max number of retries.
        api_key: Optional[str]
            Qwen QwQ Thingking API key. If not passed in will be read from env var DASHSCOPE_API_KEY.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_qwq import ChatQwQ

            llm = ChatQwQ(
                model="...",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                # api_key="...",
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk.text(), end="")

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

    Async:
        .. code-block:: python

            # Basic async invocation
            result = await llm.ainvoke(messages)

            # Access content and reasoning
            content = result.content
            reasoning = result.additional_kwargs.get("reasoning_content", "")

            # Stream response chunks
            async for chunk in await llm.astream(messages):
                print(chunk.content, end="")
                # Access reasoning in each chunk
                reasoning_chunk = chunk.additional_kwargs.get("reasoning_content", "")

            # Process tool calls in completion
            if hasattr(result, "tool_calls") and result.tool_calls:
                for tool_call in result.tool_calls:
                    tool_id = tool_call.get("id")
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args")
                    # Process tool call...

            # Batch processing of multiple message sets
            results = await llm.abatch([messages1, messages2])

    """  # noqa: E501

    model_name: str = Field(default="qwq-plus", alias="model")

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-qwq"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"api_key": "DASHSCOPE_API_KEY"}

    def _check_need_stream(self) -> bool:
        return True

    def _support_tool_choice(self) -> bool:
        return False

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: Type,
        base_generation_info: Optional[Dict],
    ) -> Optional[ChatGenerationChunk]:
        # Let parent handle tool_call_chunks properly for streaming
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )

        # Only add reasoning_content, don't interfere with tool_calls
        if (choices := chunk.get("choices")) and generation_chunk:
            top = choices[0]
            if isinstance(generation_chunk.message, AIMessageChunk):
                if delta := top.get("delta", {}):
                    if reasoning_content := delta.get("reasoning_content"):
                        generation_chunk.message.additional_kwargs[
                            "reasoning_content"
                        ] = reasoning_content

        return generation_chunk

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        kwargs["stream_options"] = {"include_usage": True}
        # Let parent handle streaming without modifications
        for chunk in super()._stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            # Fix tool_call_chunks' args to prevent invalid_tool_calls
            if (
                hasattr(chunk.message, "tool_call_chunks")
                and chunk.message.tool_call_chunks
            ):
                for tc_chunk in chunk.message.tool_call_chunks:
                    if (
                        "args" in tc_chunk
                        and isinstance(tc_chunk["args"], str)
                        and tc_chunk["args"]
                    ):
                        # Only try to fix if args looks complete (ends with })
                        if tc_chunk["args"].rstrip().endswith("}"):
                            try:
                                json.loads(tc_chunk["args"])  # If valid, leave it
                            except (JSONDecodeError, ValueError):
                                try:
                                    # Repair malformed JSON and convert back to string
                                    parsed = json_repair.loads(tc_chunk["args"])
                                    tc_chunk["args"] = json.dumps(parsed)
                                except Exception:
                                    pass  # Leave as-is if repair fails

            # Post-process last chunk to also repair invalid_tool_calls
            is_final = chunk.generation_info and chunk.generation_info.get(
                "finish_reason"
            )
            if (
                is_final
                and hasattr(chunk.message, "invalid_tool_calls")
                and chunk.message.invalid_tool_calls
            ):
                if not hasattr(chunk.message, "tool_calls"):
                    chunk.message.tool_calls = []

                for invalid_tc in chunk.message.invalid_tool_calls:
                    args_value = invalid_tc.get("args")
                    if isinstance(args_value, str):
                        try:
                            parsed_args = json.loads(args_value)
                        except (JSONDecodeError, ValueError):
                            try:
                                parsed_args = json_repair.loads(args_value)
                            except Exception:
                                continue
                    else:
                        parsed_args = args_value if args_value else {}

                    chunk.message.tool_calls.append(
                        {
                            "id": invalid_tc.get("id", ""),
                            "name": invalid_tc.get("name", ""),
                            "args": parsed_args,
                            "type": "tool_call",
                        }
                    )

                chunk.message.invalid_tool_calls = []

            yield chunk

    def generate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Any = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Override generate to repair invalid_tool_calls from malformed JSON."""
        result = super().generate(
            messages,
            stop,
            callbacks,
            tags=tags,
            metadata=metadata,
            run_name=run_name,
            run_id=run_id,
            **kwargs,
        )

        # Post-process each generation to repair invalid_tool_calls
        for generation_list in result.generations:
            for generation in generation_list:
                if (
                    hasattr(generation.message, "invalid_tool_calls")
                    and generation.message.invalid_tool_calls
                ):
                    # Repair invalid_tool_calls and add to tool_calls
                    if not hasattr(generation.message, "tool_calls"):
                        generation.message.tool_calls = []

                    for invalid_tc in generation.message.invalid_tool_calls:
                        args_value = invalid_tc.get("args")
                        if isinstance(args_value, str):
                            try:
                                parsed_args = json.loads(args_value)
                            except (JSONDecodeError, ValueError):
                                try:
                                    parsed_args = json_repair.loads(args_value)
                                except Exception:
                                    continue
                        else:
                            parsed_args = args_value if args_value else {}

                        generation.message.tool_calls.append(
                            {
                                "id": invalid_tc.get("id", ""),
                                "name": invalid_tc.get("name", ""),
                                "args": parsed_args,
                                "type": "tool_call",
                            }
                        )

                    # Clear invalid_tool_calls after processing
                    generation.message.invalid_tool_calls = []

        return result

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            # Accumulate chunks to get the final message with all tool calls
            stream = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            first_chunk = next(stream)
            accumulated = first_chunk
            for chunk in stream:
                accumulated += chunk  # type: ignore

            final_message = accumulated.message
            msg_content = final_message.content
            content = msg_content if isinstance(msg_content, str) else ""
            reasoning_content = final_message.additional_kwargs.get(
                "reasoning_content", ""
            )

            # Build tool_calls from tool_call_chunks and repair invalid_tool_calls
            tool_calls: List[Dict[str, Any]] = []

            has_chunks = (
                hasattr(final_message, "tool_call_chunks")
                and final_message.tool_call_chunks
            )
            if has_chunks:
                # Group tool_call_chunks by index
                chunks_by_index: Dict[int, Dict[str, Any]] = {}
                for tc_chunk in final_message.tool_call_chunks:
                    idx = tc_chunk.get("index", 0)
                    if idx not in chunks_by_index:
                        chunks_by_index[idx] = {
                            "id": "",
                            "name": "",
                            "args": "",
                        }
                    if tc_chunk.get("id"):
                        chunks_by_index[idx]["id"] = tc_chunk["id"]
                    if tc_chunk.get("name"):
                        chunks_by_index[idx]["name"] = tc_chunk["name"]
                    if tc_chunk.get("args"):
                        chunks_by_index[idx]["args"] += tc_chunk["args"]

                # Convert to tool_calls format with JSON repair
                for tc_data in chunks_by_index.values():
                    args_str = tc_data["args"]
                    if not args_str:
                        parsed_args = {}
                    else:
                        try:
                            parsed_args = json.loads(args_str)
                        except (JSONDecodeError, ValueError):
                            # Fallback to json_repair for malformed JSON from API
                            parsed_args = json_repair.loads(args_str)

                    tool_calls.append(
                        {
                            "id": tc_data["id"],
                            "name": tc_data["name"],
                            "args": parsed_args,
                            "type": "function",
                        }
                    )

            # Also repair invalid_tool_calls that failed parent's JSON parsing
            if (
                hasattr(final_message, "invalid_tool_calls")
                and final_message.invalid_tool_calls
            ):
                for invalid_tc in final_message.invalid_tool_calls:
                    args_value = invalid_tc.get("args")
                    if isinstance(args_value, str):
                        # Try to repair the malformed JSON string
                        try:
                            parsed_args = json.loads(args_value)
                        except (JSONDecodeError, ValueError):
                            try:
                                parsed_args = json_repair.loads(args_value)
                            except Exception:
                                # If repair fails, skip this tool call
                                continue
                    else:
                        parsed_args = args_value if args_value else {}

                    tool_calls.append(
                        {
                            "id": invalid_tc.get("id", ""),
                            "name": invalid_tc.get("name", ""),
                            "args": parsed_args,
                            "type": "function",
                        }
                    )

            # Extract usage info from the accumulated chunk's generation_info
            generation_info = accumulated.generation_info or {}
            # Extract usage metadata from accumulated message if available
            if hasattr(final_message, "usage_metadata"):
                usage_metadata = final_message.usage_metadata  # type: ignore
            else:
                usage_metadata = {}

            return ChatResult(
                generations=[
                    ChatGeneration(
                        generation_info=generation_info,
                        message=AIMessage(
                            content=content,
                            additional_kwargs={"reasoning_content": reasoning_content},
                            tool_calls=tool_calls,
                            usage_metadata=usage_metadata,
                            response_metadata={"model_name": self.model_name},
                        ),
                    )
                ],
                # Explicitly include usage at the ChatResult level if needed
                llm_output=(
                    {"usage": generation_info.get("usage")}
                    if "usage" in generation_info
                    else None
                ),
            )

        except JSONDecodeError as e:
            raise JSONDecodeError(
                "Qwen QwQ Thingking API returned an invalid response. "
                "Please check the API status and try again.",
                e.doc,
                e.pos,
            ) from e

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            # Accumulate chunks to get the final message with all tool calls
            accumulated = None
            async for chunk in self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                if accumulated is None:
                    accumulated = chunk
                else:
                    accumulated += chunk  # type: ignore

            if accumulated is None:
                raise ValueError("No chunks received from stream")

            final_message = accumulated.message
            msg_content = final_message.content
            content = msg_content if isinstance(msg_content, str) else ""
            reasoning_content = final_message.additional_kwargs.get(
                "reasoning_content", ""
            )

            # Build tool_calls from tool_call_chunks and repair invalid_tool_calls
            tool_calls: List[Dict[str, Any]] = []

            has_chunks = (
                hasattr(final_message, "tool_call_chunks")
                and final_message.tool_call_chunks
            )
            if has_chunks:
                # Group tool_call_chunks by index
                chunks_by_index: Dict[int, Dict[str, Any]] = {}
                for tc_chunk in final_message.tool_call_chunks:
                    idx = tc_chunk.get("index", 0)
                    if idx not in chunks_by_index:
                        chunks_by_index[idx] = {
                            "id": "",
                            "name": "",
                            "args": "",
                        }
                    if tc_chunk.get("id"):
                        chunks_by_index[idx]["id"] = tc_chunk["id"]
                    if tc_chunk.get("name"):
                        chunks_by_index[idx]["name"] = tc_chunk["name"]
                    if tc_chunk.get("args"):
                        chunks_by_index[idx]["args"] += tc_chunk["args"]

                # Convert to tool_calls format with JSON repair
                for tc_data in chunks_by_index.values():
                    args_str = tc_data["args"]
                    if not args_str:
                        parsed_args = {}
                    else:
                        try:
                            parsed_args = json.loads(args_str)
                        except (JSONDecodeError, ValueError):
                            # Fallback to json_repair for malformed JSON from API
                            parsed_args = json_repair.loads(args_str)

                    tool_calls.append(
                        {
                            "id": tc_data["id"],
                            "name": tc_data["name"],
                            "args": parsed_args,
                            "type": "function",
                        }
                    )

            # Also repair invalid_tool_calls that failed parent's JSON parsing
            if (
                hasattr(final_message, "invalid_tool_calls")
                and final_message.invalid_tool_calls
            ):
                for invalid_tc in final_message.invalid_tool_calls:
                    args_value = invalid_tc.get("args")
                    if isinstance(args_value, str):
                        # Try to repair the malformed JSON string
                        try:
                            parsed_args = json.loads(args_value)
                        except (JSONDecodeError, ValueError):
                            try:
                                parsed_args = json_repair.loads(args_value)
                            except Exception:
                                # If repair fails, skip this tool call
                                continue
                    else:
                        parsed_args = args_value if args_value else {}

                    tool_calls.append(
                        {
                            "id": invalid_tc.get("id", ""),
                            "name": invalid_tc.get("name", ""),
                            "args": parsed_args,
                            "type": "function",
                        }
                    )

            # Extract usage info from the accumulated chunk's generation_info
            generation_info = accumulated.generation_info or {}
            # Extract usage metadata from accumulated message if available
            if hasattr(final_message, "usage_metadata"):
                usage_metadata = final_message.usage_metadata  # type: ignore
            else:
                usage_metadata = {}

            return ChatResult(
                generations=[
                    ChatGeneration(
                        generation_info=generation_info,
                        message=AIMessage(
                            content=content,
                            additional_kwargs={"reasoning_content": reasoning_content},
                            tool_calls=tool_calls,
                            usage_metadata=usage_metadata,
                            response_metadata={"model_name": self.model_name},
                        ),
                    )
                ],
                # Explicitly include usage at the ChatResult level if needed
                llm_output=(
                    {"usage": generation_info.get("usage")}
                    if "usage" in generation_info
                    else None
                ),
            )

        except JSONDecodeError as e:
            raise JSONDecodeError(
                "Qwen QwQ Thingking API returned an invalid response. "
                "Please check the API status and try again.",
                e.doc,
                e.pos,
            ) from e

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        kwargs["stream_options"] = {"include_usage": True}
        # Let parent handle async streaming without modifications
        async for chunk in super()._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            # Fix tool_call_chunks' args to prevent invalid_tool_calls
            if (
                hasattr(chunk.message, "tool_call_chunks")
                and chunk.message.tool_call_chunks
            ):
                for tc_chunk in chunk.message.tool_call_chunks:
                    if (
                        "args" in tc_chunk
                        and isinstance(tc_chunk["args"], str)
                        and tc_chunk["args"]
                    ):
                        # Only try to fix if args looks complete (ends with })
                        if tc_chunk["args"].rstrip().endswith("}"):
                            try:
                                json.loads(tc_chunk["args"])  # If valid, leave it
                            except (JSONDecodeError, ValueError):
                                try:
                                    # Repair malformed JSON and convert back to string
                                    parsed = json_repair.loads(tc_chunk["args"])
                                    tc_chunk["args"] = json.dumps(parsed)
                                except Exception:
                                    pass  # Leave as-is if repair fails

            # Post-process last chunk to also repair invalid_tool_calls
            is_final = chunk.generation_info and chunk.generation_info.get(
                "finish_reason"
            )
            if (
                is_final
                and hasattr(chunk.message, "invalid_tool_calls")
                and chunk.message.invalid_tool_calls
            ):
                if not hasattr(chunk.message, "tool_calls"):
                    chunk.message.tool_calls = []

                for invalid_tc in chunk.message.invalid_tool_calls:
                    args_value = invalid_tc.get("args")
                    if isinstance(args_value, str):
                        try:
                            parsed_args = json.loads(args_value)
                        except (JSONDecodeError, ValueError):
                            try:
                                parsed_args = json_repair.loads(args_value)
                            except Exception:
                                continue
                    else:
                        parsed_args = args_value if args_value else {}

                    chunk.message.tool_calls.append(
                        {
                            "id": invalid_tc.get("id", ""),
                            "name": invalid_tc.get("name", ""),
                            "args": parsed_args,
                            "type": "tool_call",
                        }
                    )

                chunk.message.invalid_tool_calls = []

            yield chunk

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "json_mode",
        include_raw: bool = False,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        (
            """Create a version of this chat model that returns outputs formatted """
            """according to the given schema.

        Args:
            schema: The schema to use for formatting the output. Can be a dictionary"""
            """or a Pydantic model.
            include_raw: Whether to include the raw model output in the output.
            method: The method to use for formatting the output.
            strict: Whether to enforce strict validation of the output against """
            """the schema. If not provided, will default to True.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            A runnable that returns outputs formatted according to the given schema.
        """
        )

        if method != "function_calling":
            method = "function_calling"

        return super().with_structured_output(
            method=method,
            schema=schema,
            include_raw=include_raw,
            strict=strict,
            **kwargs,
        )


class ChatQwen(_BaseChatQwen):
    """Qwen series models integration, specifically strengthen for Qwen3 API access.

    Setup:
        Install ``langchain-qwq`` and set environment variable ``DASHSCOPE_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-qwq
            export DASHSCOPE_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of Qwen model to use, e.g. "qwen3-32b".
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.

        enable_thinking: Optional[bool]
            Whether to enable thinking.
        thinking_budget: Optional[int]
            Thinking budget.

    Key init args — client params:
        timeout: Optional[float]
            Timeout for requests.
        max_retries: int
            Max number of retries.
        api_key: Optional[str]
            Qwen QwQ Thingking API key. If not passed in will be read from env var DASHSCOPE_API_KEY.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_qwq import ChatQwen

            llm = ChatQwen(
                model="qwen3-32b",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                enable_thinking=True,
                thinking_budget=100,
                # api_key="...",
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk.text(), end="")

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

    Async:
        .. code-block:: python

            # Basic async invocation
            result = await llm.ainvoke(messages)

            # Access content and reasoning
            content = result.content
            reasoning = result.additional_kwargs.get("reasoning_content", "")

            # Stream response chunks
            async for chunk in await llm.astream(messages):
                print(chunk.content, end="")
                # Access reasoning in each chunk
                reasoning_chunk = chunk.additional_kwargs.get("reasoning_content", "")

            # Process tool calls in completion
            if hasattr(result, "tool_calls") and result.tool_calls:
                for tool_call in result.tool_calls:
                    tool_id = tool_call.get("id")
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args")
                    # Process tool call...

            # Batch processing of multiple message sets
            results = await llm.abatch([messages1, messages2])

    Enable Thinking (Qwen3 model only):
        .. code-block:: python

            # Enable thinking with budget control
            llm = ChatQwen(
                model="qwen3-8b",
                enable_thinking=True,
                thinking_budget=100  # Set thinking steps limit
            )

            # Check thinking process in response
            result = llm.invoke("Explain quantum computing")
    """  # noqa: E501

    model_name: str = Field(default="qwen3-32b", alias="model")
    """The name of the model"""

    enable_thinking: Optional[bool] = Field(default=None)
    """Whether to enable thinking"""

    thinking_budget: Optional[int] = Field(default=None)
    """Thinking budget"""

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-qwen"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling ChatQwen API."""
        if self.enable_thinking is not None:
            if self.extra_body is None:
                self.extra_body = {"enable_thinking": self.enable_thinking}
            else:
                self.extra_body = {
                    **self.extra_body,
                    "enable_thinking": self.enable_thinking,
                }

        if self.thinking_budget is not None:
            if self.extra_body is None:
                self.extra_body = {"thinking_budget": self.thinking_budget}
            else:
                self.extra_body = {
                    **self.extra_body,
                    "thinking_budget": self.thinking_budget,
                }

        params = super()._default_params

        return params

    def _is_open_source_model(self) -> bool:
        import re

        pattern = r"\d+b"
        return bool(re.search(pattern, self.model_name.lower()))

    def _is_thinking_model(self) -> bool:
        is_open_source_model = self._is_open_source_model()
        if is_open_source_model:
            if (
                self.model_name.startswith("qwen3")
                and "instruct" not in self.model_name
                and self.enable_thinking is not False
            ) or "thinking" in self.model_name:
                return True
            return False
        return self.enable_thinking is True

    def _check_need_stream(self) -> bool:
        api_base = self.api_base or ""

        if "dashscope" in api_base and self._is_thinking_model():
            return True

        return False

    def _support_tool_choice(self) -> bool:
        thinking_model = self._is_thinking_model()
        if "thinking" not in self.model_name:
            self.enable_thinking = False
            return True
        return not thinking_model

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: Type,
        base_generation_info: Optional[Dict],
    ) -> Optional[ChatGenerationChunk]:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if (choices := chunk.get("choices")) and generation_chunk:
            top = choices[0]
            if isinstance(generation_chunk.message, AIMessageChunk):
                if reasoning_content := top.get("delta", {}).get("reasoning_content"):
                    generation_chunk.message.additional_kwargs["reasoning_content"] = (
                        reasoning_content
                    )
                # Handle use via OpenRouter
                elif reasoning := top.get("delta", {}).get("reasoning"):
                    generation_chunk.message.additional_kwargs["reasoning_content"] = (
                        reasoning
                    )

        return generation_chunk

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        kwargs["stream_options"] = {"include_usage": True}
        try:
            for chunk in super()._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk
        except JSONDecodeError as e:
            raise JSONDecodeError(
                "DashScope Qwen API returned an invalid response. "
                "Please check the API status and try again.",
                e.doc,
                e.pos,
            ) from e

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        kwargs["stream_options"] = {"include_usage": True}
        try:
            async for chunk in super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk
        except JSONDecodeError as e:
            raise JSONDecodeError(
                "DashScope Qwen API returned an invalid response. "
                "Please check the API status and try again.",
                e.doc,
                e.pos,
            ) from e

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return super()._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        except JSONDecodeError as e:
            raise JSONDecodeError(
                "DashScope Qwen API returned an invalid response. "
                "Please check the API status and try again.",
                e.doc,
                e.pos,
            ) from e

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        except JSONDecodeError as e:
            raise JSONDecodeError(
                "DashScope Qwen API returned an invalid response. "
                "Please check the API status and try again.",
                e.doc,
                e.pos,
            ) from e

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        include_raw: bool = False,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema:
                The output schema. Can be passed in as:

                - an OpenAI function/tool schema,
                - a JSON Schema,
                - a TypedDict class (support added in 0.1.20),
                - or a Pydantic class.

                If ``schema`` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated. See :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`
                for more on how to properly specify types and descriptions of
                schema fields when specifying a Pydantic or TypedDict class.

            method: The method for steering model generation, one of:

                - "function_calling":
                    Uses DashScope Qwen's `tool-calling features <https://help.aliyun.com/zh/model-studio/qwen-function-calling>`_.
                - "json_mode":
                    Uses DashScope Qwen's `JSON mode feature <https://help.aliyun.com/zh/model-studio/json-mode>`_.


            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

            kwargs: Additional keyword args aren't supported.

        Returns:
            A Runnable that takes same inputs as a :class:`langchain_core.language_models.chat.BaseChatModel`.

            | If ``include_raw`` is False and ``schema`` is a Pydantic class, Runnable outputs an instance of ``schema`` (i.e., a Pydantic object). Otherwise, if ``include_raw`` is False then Runnable outputs a dict.

            | If ``include_raw`` is True, then Runnable outputs a dict with keys:

            - "raw": BaseMessage
            - "parsed": None if there was a parsing error, otherwise the type depends on the ``schema`` as described above.
            - "parsing_error": Optional[BaseException]

        """  # noqa: E501

        thinking_model = self._is_thinking_model()
        if thinking_model:
            if "thinking" not in self.model_name:
                self.enable_thinking = False
            else:
                method = "function_calling"
        return super().with_structured_output(
            schema=schema,
            method=method,
            include_raw=include_raw,
            strict=strict,
            **kwargs,
        )
