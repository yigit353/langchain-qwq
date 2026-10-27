"""Integration for QwQ and most Qwen series chat models"""

import json
from collections.abc import Mapping
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
from langchain_core.language_models.chat_models import (
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.outputs import (
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.runnables import Runnable
from pydantic import (
    Field,
)

from .base import _BaseChatQwen, _DictOrPydantic, _DictOrPydanticClass


class _ToolCallStreamBuffer:
    """Hold tool arguments until complete without delaying other stream data."""

    def __init__(self) -> None:
        self.pending: Optional[AIMessageChunk] = None

    def _take_calls(self) -> List[ToolCallChunk]:
        if self.pending is None:
            return []
        calls = self.pending.tool_call_chunks
        self.pending = None
        for call in calls:
            arguments = call["args"]
            if not arguments:
                continue
            try:
                json.loads(arguments)
            except JSONDecodeError:
                try:
                    repaired = json_repair.loads(arguments)
                except Exception:
                    continue
                if isinstance(repaired, dict):
                    call["args"] = json.dumps(repaired)
        return calls

    @staticmethod
    def _with_calls(
        message: AIMessageChunk, calls: List[ToolCallChunk]
    ) -> AIMessageChunk:
        data = message.model_dump(
            exclude={"tool_call_chunks", "tool_calls", "invalid_tool_calls"}
        )
        data["additional_kwargs"].pop("tool_calls", None)
        if calls:
            data["additional_kwargs"]["tool_calls"] = [
                {
                    "index": call["index"],
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["args"]},
                }
                for call in calls
            ]
        # Revalidate so raw, parsed and invalid tool-call fields stay consistent.
        return AIMessageChunk(**data, tool_call_chunks=calls)

    def process(self, chunk: ChatGenerationChunk) -> ChatGenerationChunk:
        message = chunk.message
        if not isinstance(message, AIMessageChunk):
            return chunk
        if message.tool_call_chunks:
            tool_chunk = AIMessageChunk(
                content="", tool_call_chunks=message.tool_call_chunks
            )
            self.pending = (
                tool_chunk if self.pending is None else self.pending + tool_chunk
            )
        finished = (chunk.generation_info or {}).get("finish_reason") is not None
        calls = self._take_calls() if finished else []
        return chunk.model_copy(update={"message": self._with_calls(message, calls)})

    def finish(self) -> Optional[ChatGenerationChunk]:
        # A clean EOF can complete providers that omit finish_reason. Exceptions
        # and cancellation do not call this, so partial calls are never emitted.
        if self.pending is None:
            return None
        return ChatGenerationChunk(
            message=self._with_calls(
                AIMessageChunk(
                    content="", response_metadata={"model_provider": "dashscope"}
                ),
                self._take_calls(),
            )
        )


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
        Text and reasoning are delivered as they arrive. Tool-call arguments
        are buffered until completion, then malformed JSON objects are repaired
        once. This applies to streaming and invocation, synchronously and
        asynchronously. Arguments that cannot be repaired remain available in
        ``tool_call_chunks`` and ``invalid_tool_calls``.

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
        buffer = _ToolCallStreamBuffer()
        for chunk in super()._stream(messages, stop=stop, run_manager=None, **kwargs):
            chunk = buffer.process(chunk)
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    logprobs=(chunk.generation_info or {}).get("logprobs"),
                )
            yield chunk
        final_chunk = buffer.finish()
        if final_chunk is not None:
            if run_manager:
                run_manager.on_llm_new_token(final_chunk.text, chunk=final_chunk)
            yield final_chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return generate_from_stream(
            self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await agenerate_from_stream(
            self._astream(messages, stop=stop, run_manager=run_manager, **kwargs)
        )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        kwargs["stream_options"] = {"include_usage": True}
        buffer = _ToolCallStreamBuffer()
        async for chunk in super()._astream(
            messages, stop=stop, run_manager=None, **kwargs
        ):
            chunk = buffer.process(chunk)
            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    logprobs=(chunk.generation_info or {}).get("logprobs"),
                )
            yield chunk
        final_chunk = buffer.finish()
        if final_chunk is not None:
            if run_manager:
                await run_manager.on_llm_new_token(final_chunk.text, chunk=final_chunk)
            yield final_chunk

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
        chat_template_kwargs = (
            self.extra_body.get("chat_template_kwargs") if self.extra_body else None
        )
        has_nested_enable_thinking = (
            isinstance(chat_template_kwargs, Mapping)
            and "enable_thinking" in chat_template_kwargs
        )
        if self.enable_thinking is not None and not has_nested_enable_thinking:
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
