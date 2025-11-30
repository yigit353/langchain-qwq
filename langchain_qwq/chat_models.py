"""Integration for QwQ and most Qwen series chat models"""

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
    cast,
)

import json_repair as json
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessageChunk, BaseMessage, ToolCall
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from pydantic import (
    Field,
)

from .base import _BaseChatQwen, _DictOrPydantic, _DictOrPydanticClass

# Store the original __add__ method
original_add = AIMessageChunk.__add__


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
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )

        if (choices := chunk.get("choices")) and generation_chunk:
            top = choices[0]
            if isinstance(generation_chunk.message, AIMessageChunk):
                if delta := top.get("delta", {}):
                    if reasoning_content := delta.get("reasoning_content"):
                        generation_chunk.message.additional_kwargs[
                            "reasoning_content"
                        ] = reasoning_content

                    # Handle tool calls
                    if tool_calls := delta.get("tool_calls"):
                        generation_chunk.message.tool_call_chunks = []
                        for tool_call in tool_calls:
                            generation_chunk.message.tool_call_chunks.append(
                                {
                                    "index": tool_call.get("index"),
                                    "id": tool_call.get("id", ""),
                                    "type": "function",  # type: ignore
                                    "name": tool_call.get("function", {}).get(
                                        "name", ""
                                    ),
                                    "args": tool_call.get("function", {}).get(
                                        "arguments", ""
                                    ),
                                }
                            )

        return generation_chunk

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        from langchain_core.messages import AIMessageChunk

        # Helper function to check if a tool call is valid
        def is_valid_tool_call(tc: ToolCall) -> bool:
            # Filter out invalid/incomplete tool calls
            if not tc:
                return False

            # Check that we have an ID
            if not tc.get("id"):
                return False

            # Check that we have a name
            if tc.get("name") is None and tc.get("type") == "function":
                return False

            # Check for valid args
            args = tc.get("args")
            if args is None or args == "}" or args == "{}}":
                return False

            return True

        # Create a patched version that ensures tool_calls are preserved
        def patched_add(self: AIMessageChunk, other: AIMessageChunk) -> AIMessageChunk:
            if not isinstance(result := original_add(self, other), AIMessageChunk):
                raise ValueError("Result is not an AIMessageChunk")

            # Ensure tool_calls are preserved across additions
            if hasattr(self, "tool_calls") and self.tool_calls:
                if not hasattr(result, "tool_calls") or not result.tool_calls:
                    result.tool_calls = [
                        tc for tc in self.tool_calls if is_valid_tool_call(tc)
                    ]

            if hasattr(other, "tool_calls") and other.tool_calls:
                if not hasattr(result, "tool_calls"):
                    result.tool_calls = [
                        tc for tc in other.tool_calls if is_valid_tool_call(tc)
                    ]
                else:
                    # Merge unique tool calls, filtering out invalid ones
                    existing_ids = {tc.get("id", "") for tc in result.tool_calls}
                    for tc in other.tool_calls:
                        if tc.get("id", "") not in existing_ids and is_valid_tool_call(
                            tc
                        ):
                            result.tool_calls.append(tc)

            # Clear invalid_tool_calls if we have valid tool_calls for the same ID
            if (
                hasattr(result, "tool_calls")
                and result.tool_calls
                and hasattr(result, "invalid_tool_calls")
                and result.invalid_tool_calls
            ):
                valid_ids = {tc.get("id") for tc in result.tool_calls if tc.get("id")}
                result.invalid_tool_calls = [
                    tc
                    for tc in result.invalid_tool_calls
                    if tc.get("id") not in valid_ids
                ]

            return result

        # Monkey patch the __add__ method
        AIMessageChunk.__add__ = patched_add  # type: ignore

        try:
            kwargs["stream_options"] = {"include_usage": True}

            # Track tool call chunks to reconstruct tool calls at the end
            accumulated_tool_call_chunks: Dict[int, Dict[str, Any]] = {}

            # Original streaming
            for chunk in super()._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                # Accumulate tool call chunks
                if (
                    isinstance(chunk.message, AIMessageChunk)
                    and chunk.message.tool_call_chunks
                ):
                    for tc_chunk in chunk.message.tool_call_chunks:
                        index = tc_chunk.get("index")
                        if index is not None:
                            if index not in accumulated_tool_call_chunks:
                                accumulated_tool_call_chunks[index] = {
                                    "index": index,
                                    "id": tc_chunk.get("id", ""),
                                    "name": tc_chunk.get("name", ""),
                                    "args": tc_chunk.get("args", "") or "",
                                    "type": "tool_call",
                                }
                            else:
                                if tc_chunk.get("id"):
                                    accumulated_tool_call_chunks[index]["id"] = (
                                        tc_chunk.get("id")
                                    )
                                if tc_chunk.get("name"):
                                    accumulated_tool_call_chunks[index]["name"] = (
                                        tc_chunk.get("name")
                                    )
                                if tc_chunk.get("args"):
                                    accumulated_tool_call_chunks[index]["args"] += (
                                        tc_chunk.get("args") or ""
                                    )

                yield chunk

            # Yield final chunk with parsed tool calls if any
            if accumulated_tool_call_chunks:
                tool_calls: List[ToolCall] = []
                for index in sorted(accumulated_tool_call_chunks.keys()):
                    tc = accumulated_tool_call_chunks[index]
                    try:
                        args = json.loads(str(tc["args"]))
                        tool_calls.append(
                            {
                                "name": str(tc["name"]),
                                "args": cast(Dict[str, Any], args),
                                "id": str(tc["id"]),
                                "type": "tool_call",
                            }
                        )
                    except Exception:
                        pass

                if tool_calls:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content="", tool_calls=tool_calls)
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
        from langchain_core.messages import AIMessageChunk

        # Helper function to check if a tool call is valid
        def is_valid_tool_call(tc: ToolCall) -> bool:
            # Filter out invalid/incomplete tool calls
            if not tc:
                return False

            # Check that we have an ID
            if not tc.get("id"):
                return False

            # Check that we have a name
            if tc.get("name") is None and tc.get("type") == "function":
                return False

            # Check for valid args
            args = tc.get("args")
            if args is None or args == "}" or args == "{}}":
                return False

            return True

        # Create a patched version that ensures tool_calls are preserved
        def patched_add(self: AIMessageChunk, other: AIMessageChunk) -> AIMessageChunk:
            if not isinstance(result := original_add(self, other), AIMessageChunk):
                raise ValueError("Result is not an AIMessageChunk")

            # Ensure tool_calls are preserved across additions
            if hasattr(self, "tool_calls") and self.tool_calls:
                if not hasattr(result, "tool_calls") or not result.tool_calls:
                    result.tool_calls = [
                        tc for tc in self.tool_calls if is_valid_tool_call(tc)
                    ]

            if hasattr(other, "tool_calls") and other.tool_calls:
                if not hasattr(result, "tool_calls"):
                    result.tool_calls = [
                        tc for tc in other.tool_calls if is_valid_tool_call(tc)
                    ]
                else:
                    # Merge unique tool calls, filtering out invalid ones
                    existing_ids = {tc.get("id", "") for tc in result.tool_calls}
                    for tc in other.tool_calls:
                        if tc.get("id", "") not in existing_ids and is_valid_tool_call(
                            tc
                        ):
                            result.tool_calls.append(tc)

            # Clear invalid_tool_calls if we have valid tool_calls for the same ID
            if (
                hasattr(result, "tool_calls")
                and result.tool_calls
                and hasattr(result, "invalid_tool_calls")
                and result.invalid_tool_calls
            ):
                valid_ids = {tc.get("id") for tc in result.tool_calls if tc.get("id")}
                result.invalid_tool_calls = [
                    tc
                    for tc in result.invalid_tool_calls
                    if tc.get("id") not in valid_ids
                ]

            return result

        # Monkey patch the __add__ method
        AIMessageChunk.__add__ = patched_add  # type: ignore

        try:
            kwargs["stream_options"] = {"include_usage": True}

            # Track tool call chunks to reconstruct tool calls at the end
            accumulated_tool_call_chunks: Dict[int, Dict[str, Any]] = {}

            # Original async streaming
            async for chunk in super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                # Accumulate tool call chunks
                if (
                    isinstance(chunk.message, AIMessageChunk)
                    and chunk.message.tool_call_chunks
                ):
                    for tc_chunk in chunk.message.tool_call_chunks:
                        index = tc_chunk.get("index")
                        if index is not None:
                            if index not in accumulated_tool_call_chunks:
                                accumulated_tool_call_chunks[index] = {
                                    "index": index,
                                    "id": tc_chunk.get("id", ""),
                                    "name": tc_chunk.get("name", ""),
                                    "args": tc_chunk.get("args", "") or "",
                                    "type": "tool_call",
                                }
                            else:
                                if tc_chunk.get("id"):
                                    accumulated_tool_call_chunks[index]["id"] = (
                                        tc_chunk.get("id")
                                    )
                                if tc_chunk.get("name"):
                                    accumulated_tool_call_chunks[index]["name"] = (
                                        tc_chunk.get("name")
                                    )
                                if tc_chunk.get("args"):
                                    accumulated_tool_call_chunks[index]["args"] += (
                                        tc_chunk.get("args") or ""
                                    )

                yield chunk

            # Yield final chunk with parsed tool calls if any
            if accumulated_tool_call_chunks:
                tool_calls: List[ToolCall] = []
                for index in sorted(accumulated_tool_call_chunks.keys()):
                    tc = accumulated_tool_call_chunks[index]
                    try:
                        args = json.loads(str(tc["args"]))
                        tool_calls.append(
                            {
                                "name": str(tc["name"]),
                                "args": cast(Dict[str, Any], args),
                                "id": str(tc["id"]),
                                "type": "tool_call",
                            }
                        )
                    except Exception:
                        pass

                if tool_calls:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content="", tool_calls=tool_calls)
                    )

        finally:
            # Restore the original method
            AIMessageChunk.__add__ = original_add  # type: ignore

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
