"""Integration for Qwen QwQ and Qwen3  with thinking chat models."""

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
    TypeVar,
    Union,
)

import json_repair as json
import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils import from_env, secret_from_env
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

DEFAULT_API_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[Dict[str, Any], Type[_BM], Type]
_DictOrPydantic = Union[Dict, _BM]

# Store the original __add__ method
original_add = AIMessageChunk.__add__


class ChatQwQ(BaseChatOpenAI):
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
    """The name of the model"""
    api_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("DASHSCOPE_API_KEY", default=None)
    )
    """Qwen QwQ Thinking API key"""
    api_base: str = Field(
        default_factory=from_env("DASHSCOPE_API_BASE", default=DEFAULT_API_BASE),
        alias="base_url",
    )
    """Qwen QwQ Thinking API base URL"""

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-qwq"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"api_key": "DASHSCOPE_API_KEY"}

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        if self.api_base == DEFAULT_API_BASE and not (
            self.api_key and self.api_key.get_secret_value()
        ):
            raise ValueError(
                "If using default api base, DASHSCOPE_API_KEY must be set."
            )
        client_params: dict = {
            k: v
            for k, v in {
                "api_key": self.api_key.get_secret_value() if self.api_key else None,
                "base_url": self.api_base,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "default_headers": self.default_headers,
                "default_query": self.default_query,
            }.items()
            if v is not None
        }

        if not (self.client or None):  # type: ignore
            sync_specific: dict = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
            self.client = self.root_client.chat.completions
        if not (self.async_client or None):  # type: ignore
            async_specific: dict = {"http_client": self.http_async_client}
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )
            self.async_client = self.root_async_client.chat.completions
        return self

    def _create_chat_result(
        self,
        response: Union[dict, openai.BaseModel],
        generation_info: Optional[Dict] = None,
    ) -> ChatResult:
        rtn = super()._create_chat_result(response, generation_info)

        if not isinstance(response, openai.BaseModel):
            return rtn

        if hasattr(response.choices[0].message, "reasoning_content"):  # type: ignore
            rtn.generations[0].message.additional_kwargs["reasoning_content"] = (
                response.choices[0].message.reasoning_content  # type: ignore
            )
        # Handle use via OpenRouter
        elif hasattr(response.choices[0].message, "model_extra"):  # type: ignore
            model_extra = response.choices[0].message.model_extra  # type: ignore
            if isinstance(model_extra, dict) and (
                reasoning := model_extra.get("reasoning")
            ):
                rtn.generations[0].message.additional_kwargs["reasoning_content"] = (
                    reasoning
                )

        return rtn

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
                        generation_chunk.message.tool_calls = []
                        for tool_call in tool_calls:
                            generation_chunk.message.tool_calls.append(
                                {
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

            return result

        # Monkey patch the __add__ method
        AIMessageChunk.__add__ = patched_add  # type: ignore

        try:
            kwargs["stream_options"] = {"include_usage": True}
            # Original streaming
            for chunk in super()._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk
        finally:
            # Restore the original method
            AIMessageChunk.__add__ = original_add  # type: ignore

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            chunks = list(
                self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            )
            content = ""
            reasoning_content = ""
            tool_calls = []
            current_tool_calls = {}  # Track tool calls being built

            for chunk in chunks:
                if isinstance(chunk.message.content, str):
                    content += chunk.message.content
                reasoning_content += chunk.message.additional_kwargs.get(
                    "reasoning_content", ""
                )

                if chunk_tool_calls := chunk.message.additional_kwargs.get(
                    "tool_calls", []
                ):
                    for tool_call in chunk_tool_calls:
                        index = tool_call.get("index", "")

                        # Initialize tool call entry if needed
                        if index not in current_tool_calls:
                            current_tool_calls[index] = {
                                "id": "",
                                "name": "",
                                "args": "",
                                "type": "function",
                            }

                        # Update tool call ID
                        if tool_id := tool_call.get("id"):
                            current_tool_calls[index]["id"] = tool_id

                        # Update function name and arguments
                        if function := tool_call.get("function"):
                            if name := function.get("name"):
                                current_tool_calls[index]["name"] = name
                            if args := function.get("arguments"):
                                current_tool_calls[index]["args"] += args

            # Convert accumulated tool calls to final format
            tool_calls = list(current_tool_calls.values())
            for tool_call in tool_calls:
                tool_call["args"] = json.loads(tool_call["args"])

            last_chunk = chunks[-1]

            # Extract usage info from the last chunk's generation_info
            generation_info = last_chunk.generation_info or {}
            # Extract usage metadata from chunk if available
            if hasattr(last_chunk.message, "usage_metadata"):
                usage_metadata = last_chunk.message.usage_metadata  # type: ignore
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
            chunks = [
                chunk
                async for chunk in self._astream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            ]
            content = ""
            reasoning_content = ""
            tool_calls = []
            current_tool_calls = {}  # Track tool calls being built

            for chunk in chunks:
                if isinstance(chunk.message.content, str):
                    content += chunk.message.content
                reasoning_content += chunk.message.additional_kwargs.get(
                    "reasoning_content", ""
                )

                if chunk_tool_calls := chunk.message.additional_kwargs.get(
                    "tool_calls", []
                ):
                    for tool_call in chunk_tool_calls:
                        index = tool_call.get("index", "")

                        # Initialize tool call entry if needed
                        if index not in current_tool_calls:
                            current_tool_calls[index] = {
                                "id": "",
                                "name": "",
                                "args": "",
                                "type": "function",
                            }

                        # Update tool call ID
                        if tool_id := tool_call.get("id"):
                            current_tool_calls[index]["id"] = tool_id

                        # Update function name and arguments
                        if function := tool_call.get("function"):
                            if name := function.get("name"):
                                current_tool_calls[index]["name"] = name
                            if args := function.get("arguments"):
                                current_tool_calls[index]["args"] += args

            # Convert accumulated tool calls to final format
            tool_calls = list(current_tool_calls.values())
            for tool_call in tool_calls:
                tool_call["args"] = json.loads(tool_call["args"])

            last_chunk = chunks[-1]

            # Extract usage info from the last chunk's generation_info
            generation_info = last_chunk.generation_info or {}
            # Extract usage metadata from chunk if available
            if hasattr(last_chunk.message, "usage_metadata"):
                usage_metadata = last_chunk.message.usage_metadata  # type: ignore
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

            return result

        # Monkey patch the __add__ method
        AIMessageChunk.__add__ = patched_add  # type: ignore

        try:
            kwargs["stream_options"] = {"include_usage": True}
            # Original async streaming
            async for chunk in super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk
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
        import json
        import re

        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.output_parsers import BaseOutputParser
        from langchain_core.runnables import RunnableLambda
        from langchain_core.utils.function_calling import convert_to_json_schema

        if strict is None:
            strict = True

        if schema is None:
            if method != "json_mode":
                raise ValueError(
                    "schema must be provided when method is not 'json_mode'"
                )
            schema_dict = {}
            schema_name = "CustomOutput"
            output_cls = None

        else:
            # Extract schema information using convert_to_json_schema
            try:
                schema_dict = convert_to_json_schema(schema)
                if isinstance(schema, type) and is_basemodel_subclass(schema):
                    schema_name = schema.__name__
                    output_cls = schema
                elif isinstance(schema, dict):
                    schema_name = schema.get("title", "CustomOutput")
                    output_cls = None
                else:
                    schema_name = getattr(schema, "__name__", "CustomOutput")
                    output_cls = None
            except Exception:
                # Fallback for cases where convert_to_json_schema fails
                if isinstance(schema, type) and is_basemodel_subclass(schema):
                    if hasattr(schema, "model_json_schema"):
                        schema_dict = schema.model_json_schema()  # Pydantic v2
                    elif hasattr(schema, "schema"):
                        schema_dict = schema.schema()  # Pydantic v1
                    else:
                        raise ValueError(f"Unsupported Pydantic model: {schema}")
                    schema_name = schema.__name__
                    output_cls = schema
                elif isinstance(schema, dict):
                    schema_dict = schema
                    schema_name = schema_dict.get("title", "CustomOutput")
                    output_cls = None
                else:
                    raise ValueError(f"Unsupported schema type: {type(schema)}")

            if method == "function_calling":
                from operator import itemgetter

                from langchain_core.output_parsers import (
                    JsonOutputKeyToolsParser,
                    PydanticToolsParser,
                )
                from langchain_core.output_parsers.base import OutputParserLike
                from langchain_core.runnables import RunnableMap, RunnablePassthrough
                from langchain_core.utils.function_calling import convert_to_openai_tool

                is_pydantic_schema = isinstance(schema, type) and is_basemodel_subclass(
                    schema
                )
                llm = self.bind_tools([schema])
                if is_pydantic_schema:
                    output_parser: OutputParserLike = PydanticToolsParser(
                        tools=[schema],  # type: ignore[list-item]
                        first_tool_only=True,
                    )
                else:
                    key_name = convert_to_openai_tool(schema)["function"]["name"]
                    output_parser = JsonOutputKeyToolsParser(
                        key_name=key_name, first_tool_only=True
                    )

                if include_raw:
                    parser_assign = RunnablePassthrough.assign(
                        parsed=itemgetter("raw") | output_parser,
                        parsing_error=lambda _: None,
                    )
                    parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
                    parser_with_fallback = parser_assign.with_fallbacks(
                        [parser_none], exception_key="parsing_error"
                    )
                    return RunnableMap(raw=llm) | parser_with_fallback
                else:
                    return llm | output_parser

        # Create a custom output parser
        class StructuredOutputParser(BaseOutputParser[Any]):
            """Parser for structured output from QwQ."""

            def parse(self, text: str) -> Any:
                """Parse the output and convert to the target format."""
                try:
                    # Try to parse as JSON
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        # Try with regex
                        json_match = re.search(r"(\{.*\})", text, re.DOTALL)
                        if json_match:
                            try:
                                parsed = json.loads(json_match.group(1))
                            except json.JSONDecodeError:
                                # Try JSON repair
                                import json_repair as json_repair_lib

                                parsed = json_repair_lib.loads(json_match.group(1))
                        else:
                            # Try JSON repair on whole text
                            import json_repair as json_repair_lib

                            parsed = json_repair_lib.loads(text)

                    # Validate with Pydantic if needed
                    if output_cls and is_basemodel_subclass(output_cls):
                        if hasattr(output_cls, "model_validate"):
                            return output_cls.model_validate(parsed)
                        elif hasattr(output_cls, "parse_obj"):
                            return output_cls.parse_obj(parsed)
                        else:
                            raise ValueError("Unsupported Pydantic validation method")
                    return parsed
                except Exception as e:
                    if strict:
                        raise ValueError(f"Failed to parse output: {str(e)}")
                    return text

        # Create system prompt for JSON output
        system_template = (
            """You are a helpful assistant that always responds with"""
            """JSON that matches this schema:
    ```json
    {schema}
    ```

    Follow these rules:
    1. Your entire response must be valid JSON that adheres to the schema above
    2. Do not include any explanations, preambles, or text outside the JSON
    3. Do not include markdown formatting such as ```json or ``` around the JSON
    4. Make sure all required fields in the schema are included
    5. Use the correct data types for each field as specified in the schema
    6. If boolean values are required, use true or false (lowercase without quotes)
    7. If integer values are required, don't use quotes around them

    Example of a good response format:
    {{"key1": "value1", "key2": 42, "key3": false}}
    """
        )

        # Format the schema for the prompt
        formatted_schema = json.dumps(schema_dict, indent=2)
        system_content = system_template.format(schema=formatted_schema)

        # Create a function to prepare messages with the system prompt
        def prepare_messages(input_value: Any) -> List[BaseMessage]:
            """Prepare messages with system prompt for structured output."""
            if isinstance(input_value, str):
                return [
                    SystemMessage(content=system_content),
                    HumanMessage(content=input_value),
                ]
            elif isinstance(input_value, list):
                # Check if there's already a system message
                has_system = any(
                    getattr(msg, "type", None) == "system" for msg in input_value
                )
                if has_system:
                    # Modify existing system message
                    messages = input_value.copy()
                    for i, msg in enumerate(messages):
                        if getattr(msg, "type", None) == "system":
                            messages[i] = SystemMessage(
                                content=f"{msg.content}\n\n{system_content}"
                            )
                            break
                    return messages
                else:
                    # Add system message at the beginning
                    return [SystemMessage(content=system_content)] + input_value  # type: ignore
            else:
                # Convert to string and use as human message
                return [
                    SystemMessage(content=system_content),
                    HumanMessage(content=str(input_value)),
                ]

        # Create a modified version of the model with structured output format
        structured_model = self.bind(
            ls_structured_output_format={
                "schema": schema_dict,
                "name": schema_name,
                "method": method,
                "kwargs": {"method": method, "strict": strict},
            }
        )

        # Create the output parser
        output_parser = StructuredOutputParser()

        # Build the chain
        if include_raw:
            # Include raw output in the result
            def process_with_raw(x: Any) -> Dict[str, Any]:
                raw_output = structured_model.invoke(prepare_messages(x))
                try:
                    if isinstance(raw_output.content, str):
                        parsed = output_parser.parse(raw_output.content)
                    else:
                        parsed = raw_output.content
                    return {"raw": raw_output, "parsed": parsed, "parsing_error": None}
                except Exception as e:
                    return {"raw": raw_output, "parsed": None, "parsing_error": e}

            async def aprocess_with_raw(x: Any) -> Dict[str, Any]:
                raw_output = await structured_model.ainvoke(prepare_messages(x))
                try:
                    if isinstance(raw_output.content, str):
                        parsed = output_parser.parse(raw_output.content)
                    else:
                        parsed = raw_output.content
                    return {"raw": raw_output, "parsed": parsed, "parsing_error": None}
                except Exception as e:
                    return {"raw": raw_output, "parsed": None, "parsing_error": e}

            chain = RunnableLambda(process_with_raw, afunc=aprocess_with_raw)
        else:
            # Only return parsed output
            def process_without_raw(x: Any) -> Any:
                raw_output = structured_model.invoke(prepare_messages(x))
                if isinstance(raw_output.content, str):
                    return output_parser.parse(raw_output.content)
                else:
                    return raw_output.content

            async def aprocess_without_raw(x: Any) -> Any:
                raw_output = await structured_model.ainvoke(prepare_messages(x))
                if isinstance(raw_output.content, str):
                    return output_parser.parse(raw_output.content)
                else:
                    return raw_output.content

            chain = RunnableLambda(process_without_raw, afunc=aprocess_without_raw)

        return chain


class ChatQwen(ChatQwQ):
    """Qwen Qwen3 Thinking chat model integration to access models hosted in Qwen Qwen3's API.

    Setup:
        Install ``langchain-qwq`` and set environment variable ``DASHSCOPE_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-qwq
            export DASHSCOPE_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of Qwen Qwen3 model to use, e.g. "qwen3-32b".
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

            from langchain_qwq import ChatQwen

            llm = ChatQwen(
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
        params = super()._default_params
        if self.enable_thinking is not None:
            if "extra_body" not in params:
                params["extra_body"] = {}
            params["extra_body"]["enable_thinking"] = self.enable_thinking
        if self.thinking_budget is not None:
            if "extra_body" not in params:
                params["extra_body"] = {}
            params["extra_body"]["thinking_budget"] = self.thinking_budget

        return params
