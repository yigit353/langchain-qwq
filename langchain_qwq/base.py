from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Self,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

import openai
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[Dict[str, Any], Type[_BM], Type]


_DictOrPydantic = Union[Dict, _BM]


DEFAULT_API_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class _BaseChatQwen(BaseChatOpenAI):
    api_key: Optional[SecretStr] = Field(
        default_factory=secret_from_env("DASHSCOPE_API_KEY", default=None)
    )
    """Qwen API key"""
    api_base: str = Field(
        default_factory=from_env("DASHSCOPE_API_BASE", default=DEFAULT_API_BASE),
        alias="base_url",
    )
    """Qwen API base URL"""

    model_config = ConfigDict(populate_by_name=True)

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"api_key": "DASHSCOPE_API_KEY"}

    def _is_thinking_model(self) -> bool:
        return True

    def _check_need_stream(self) -> bool:
        return self._is_thinking_model()

    def _support_tool_choice(self) -> bool:
        return True

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
        if self._check_need_stream():
            self.streaming = True
        return self

    def _create_chat_result(
        self,
        response: Union[dict, openai.BaseModel],
        generation_info: Optional[Dict] = None,
    ) -> ChatResult:
        rtn = super()._create_chat_result(response, generation_info)

        if not isinstance(response, openai.BaseModel):
            return rtn

        for generation in rtn.generations:
            if generation.message.response_metadata is None:
                generation.message.response_metadata = {}
            generation.message.response_metadata["model_provider"] = "dashscope"

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
        default_chunk_class: type,
        base_generation_info: Optional[dict],
    ) -> Optional[ChatGenerationChunk]:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if not generation_chunk:
            return generation_chunk

        message_chunk = generation_chunk.message

        message_chunk.response_metadata = {
            **message_chunk.response_metadata,
            "model_provider": "dashscope",
        }
        return ChatGenerationChunk(
            message=message_chunk,
            generation_info=generation_chunk.generation_info,
        )

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict:
        payload = self._filter_disabled_params(
            **super()._get_request_payload(input_, stop=stop, **kwargs)
        )

        return payload

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, Literal["auto", "none"], bool]] = None,
        strict: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call. Options are:
                - str of the form ``"<<tool_name>>"``: calls <<tool_name>> tool.
                - ``"auto"``: automatically selects a tool (including no tool).
                - ``"none"``: does not call a tool.
                - dict of the form ``{"type": "function", "function": {"name": <<tool_name>>}}``: calls <<tool_name>> tool.
                - ``False`` or ``None``: no effect, default OpenAI behavior.
            strict: If True, model output is guaranteed to exactly match the JSON Schema
                provided in the tool definition. If True, the input schema will be
                validated according to
                https://platform.openai.com/docs/guides/structured-outputs/supported-schemas.
                If False, input schema will not be validated and model output will not
                be validated.
                If None, ``strict`` argument will not be passed to the model.
            parallel_tool_calls: Set to ``False`` to disable parallel tool use.
                Defaults to ``None`` (no specification, which allows parallel tool use).
            kwargs: Any additional parameters are passed directly to
                :meth:`~langchain_openai.chat_models.base.ChatOpenAI.bind`.

        .. versionchanged:: 0.1.21

            Support for ``strict`` argument added.

        """  # noqa: E501

        if parallel_tool_calls is None:
            kwargs["parallel_tool_calls"] = True

        if tool_choice:
            if tool_choice == "required" or tool_choice == "any":
                tool_choice = "auto"
            kwargs["tool_choice"] = tool_choice
        return super().bind_tools(tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal[
            "function_calling",
            "json_mode",
            "json_schema",
        ] = "function_calling",
        include_raw: bool = False,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        import json
        from operator import itemgetter

        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.output_parsers import (
            JsonOutputKeyToolsParser,
            JsonOutputParser,
            PydanticOutputParser,
            PydanticToolsParser,
        )
        from langchain_core.prompt_values import ChatPromptValue
        from langchain_core.runnables import (
            RunnableLambda,
            RunnableMap,
            RunnablePassthrough,
        )
        from langchain_core.utils.function_calling import (
            convert_to_json_schema,
            convert_to_openai_tool,
        )
        from langchain_openai.chat_models.base import _is_pydantic_class

        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")  # noqa: EM102

        is_pydantic_schema = _is_pydantic_class(schema)

        if method == "json_schema":
            method = "json_mode"

        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "Schema must be provided when using `method`='function_calling'"
                )

            tool_name = convert_to_openai_tool(schema)["function"]["name"]

            tool_choice = self._support_tool_choice()

            if tool_choice:
                bind_kwargs = self._filter_disabled_params(
                    parallel_tool_calls=False,
                    tool_choice=tool_name,
                    strict=strict,
                    ls_structured_output_format={
                        "kwargs": {"method": method, "strict": strict},
                        "schema": schema,
                    },
                )
            else:
                bind_kwargs = self._filter_disabled_params(
                    parallel_tool_calls=False,
                    strict=strict,
                    ls_structured_output_format={
                        "kwargs": {"method": method, "strict": strict},
                        "schema": schema,
                    },
                )

            llm = self.bind_tools([schema], **bind_kwargs)

            output_parser = (
                PydanticToolsParser(tools=[schema], first_tool_only=True)  # type: ignore
                if is_pydantic_schema
                else JsonOutputKeyToolsParser(key_name=tool_name, first_tool_only=True)
            )

            if include_raw:
                parser_assign = RunnablePassthrough.assign(
                    parsed=itemgetter("raw") | output_parser,
                    parsing_error=lambda _: None,
                )
                parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
                parser_with_fallback = parser_assign.with_fallbacks(
                    [parser_none],
                    exception_key="parsing_error",
                )
                chain = RunnableMap(raw=llm) | parser_with_fallback
            else:
                chain = llm | output_parser

        else:
            if schema and (isinstance(schema, dict) or is_pydantic_schema):
                schema_dict = convert_to_json_schema(schema)
            else:
                schema_dict = schema  # type: ignore

            # Format the schema for the prompt
            formatted_schema = json.dumps(schema_dict, indent=2)

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

            system_content = system_template.format(schema=formatted_schema)

            # Create a function to prepare messages with the system prompt
            def prepare_messages(input_value: Any) -> List[BaseMessage]:
                """Prepare messages with system prompt for structured output."""
                if isinstance(input_value, ChatPromptValue):
                    input_value = input_value.to_messages()

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

            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type, assignment]
                if is_pydantic_schema
                else JsonOutputParser()
            )

            llm = self.bind(
                response_format={"type": "json_object"},
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": schema,
                },
            )

            def parse_output(input_: AIMessage) -> Any:
                if isinstance(input_.content, str):
                    return output_parser.parse(input_.content)
                else:
                    return input_.content

            if include_raw:
                # Include raw output in the result
                def process_with_raw(x: Any) -> Dict[str, Any]:
                    raw_output = cast(AIMessage, llm.invoke(prepare_messages(x)))
                    try:
                        parsed = parse_output(raw_output)
                        return {
                            "raw": raw_output,
                            "parsed": parsed,
                            "parsing_error": None,
                        }
                    except Exception as e:
                        return {
                            "raw": raw_output,
                            "parsed": None,
                            "parsing_error": e,
                        }

                async def aprocess_with_raw(x: Any) -> Dict[str, Any]:
                    raw_output = cast(AIMessage, await llm.ainvoke(prepare_messages(x)))
                    try:
                        parsed = parse_output(raw_output)
                        return {
                            "raw": raw_output,
                            "parsed": parsed,
                            "parsing_error": None,
                        }
                    except Exception as e:
                        return {
                            "raw": raw_output,
                            "parsed": None,
                            "parsing_error": e,
                        }

                chain = RunnableLambda(process_with_raw, afunc=aprocess_with_raw)  # type: ignore
            else:
                # Only return parsed output
                def process_without_raw(x: Any) -> Any:
                    raw_output = cast(AIMessage, llm.invoke(prepare_messages(x)))
                    output = parse_output(raw_output)
                    return output

                async def aprocess_without_raw(x: Any) -> Any:
                    raw_output = cast(AIMessage, await llm.ainvoke(prepare_messages(x)))
                    output = parse_output(raw_output)
                    return output

                chain = RunnableLambda(process_without_raw, afunc=aprocess_without_raw)  # type: ignore

        return chain
