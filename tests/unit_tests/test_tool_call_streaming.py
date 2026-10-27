"""Tool arguments must survive JSON boundaries in provider stream deltas."""

import json
from typing import Any, AsyncIterator, Iterator

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, LLMResult
from pydantic import SecretStr

from langchain_qwq import ChatQwQ


class _InterruptedStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, first_event: bytes) -> None:
        self.first_event = first_event

    def __iter__(self) -> Iterator[bytes]:
        yield self.first_event
        raise httpx.ReadError("Interrupted tool-call stream")

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.first_event
        raise httpx.ReadError("Interrupted tool-call stream")


class _Recorder(BaseCallbackHandler):
    raise_error = True

    def __init__(self) -> None:
        self.chunks: list[AIMessageChunk] = []
        self.result: LLMResult | None = None

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        chunk = kwargs["chunk"]
        assert isinstance(chunk, ChatGenerationChunk)
        assert isinstance(chunk.message, AIMessageChunk)
        self.chunks.append(chunk.message.model_copy(deep=True))

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.result = response


def _tool_response(
    request: httpx.Request,
    *,
    malformed: bool,
    multiple: bool,
    fragments: tuple[str, ...] = ('{"input":', " 3}"),
    finish: str = "separate",
    interrupt: bool = False,
) -> httpx.Response:
    if malformed:
        fragments = ("{'input':", " 3}")
    calls = [("call_test", "magic_function", fragments)]
    if multiple:
        calls.append(("call_second", "other_function", ('{"input":', " 4}")))
    deltas: list[dict[str, Any]] = []
    for offset in range(max(len(parts) for _, _, parts in calls)):
        tool_calls = []
        for index, (call_id, name, parts) in enumerate(calls):
            if offset >= len(parts):
                continue
            function = {"arguments": parts[offset]}
            call: dict[str, Any] = {"index": index, "function": function}
            if offset == 0:
                function["name"] = name
                call.update(id=call_id, type="function")
            tool_calls.append(call)
        deltas.append({"tool_calls": tool_calls})
    deltas[0].update(role="assistant", reasoning_content="Thinking.", content="Answer.")
    if finish == "separate":
        deltas.append({})
    events = [
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": (
                        "tool_calls"
                        if offset == len(deltas) - 1 and finish != "eof"
                        else None
                    ),
                }
            ],
        }
        for offset, delta in enumerate(deltas)
    ]
    events.append(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }
    )
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    if interrupt:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_InterruptedStream(f"data: {json.dumps(events[0])}\n\n".encode()),
        )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=content + "data: [DONE]\n\n",
    )


@pytest.mark.parametrize("multiple", [False, True], ids=["single", "multiple"])
@pytest.mark.parametrize(
    "stream, malformed",
    [(False, False), (True, False), (False, True)],
    ids=["invoke", "stream", "invoke-malformed"],
)
def test_tool_argument_fragments(stream: bool, malformed: bool, multiple: bool) -> None:
    result: BaseMessage
    transport = httpx.MockTransport(
        lambda request: _tool_response(request, malformed=malformed, multiple=multiple)
    )
    with httpx.Client(transport=transport) as client:
        model = ChatQwQ(api_key=SecretStr("test-key"), http_client=client)
        if stream:
            full = AIMessageChunk(content="")
            for chunk in model.stream("Call magic_function(3)"):
                full += chunk
            result = full
        else:
            result = model.invoke("Call magic_function(3)")

    assert isinstance(result, AIMessage)
    assert sorted(result.tool_calls, key=lambda call: call["name"]) == [
        {
            "name": "magic_function",
            "args": {"input": 3},
            "id": "call_test",
            "type": "tool_call",
        }
    ] + (
        [
            {
                "name": "other_function",
                "args": {"input": 4},
                "id": "call_second",
                "type": "tool_call",
            }
        ]
        if multiple
        else []
    )
    assert not result.invalid_tool_calls


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("arguments", ["[]", "null"])
def test_non_object_tool_arguments_remain_invalid(arguments: str, stream: bool) -> None:
    result: AIMessage
    transport = httpx.MockTransport(
        lambda request: _tool_response(
            request, malformed=False, multiple=False, fragments=(arguments, "")
        )
    )
    with httpx.Client(transport=transport) as client:
        model = ChatQwQ(api_key=SecretStr("test-key"), http_client=client)
        if stream:
            full = AIMessageChunk(content="")
            for chunk in model.stream("Call magic_function(3)"):
                full += chunk
            result = full
        else:
            result = model.invoke("Call magic_function(3)")

    assert isinstance(result, AIMessage)
    assert not result.tool_calls
    assert len(result.invalid_tool_calls) == 1
    invalid_call = result.invalid_tool_calls[0]
    assert invalid_call["name"] == "magic_function"
    assert invalid_call["id"] == "call_test"
    assert invalid_call["args"] == arguments


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("arguments", ["[]", "null"])
async def test_async_non_object_tool_arguments_remain_invalid(
    arguments: str, stream: bool
) -> None:
    result: AIMessage
    transport = httpx.MockTransport(
        lambda request: _tool_response(
            request, malformed=False, multiple=False, fragments=(arguments, "")
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        model = ChatQwQ(api_key=SecretStr("test-key"), http_async_client=client)
        if stream:
            full = AIMessageChunk(content="")
            async for chunk in model.astream("Call magic_function(3)"):
                full += chunk
            result = full
        else:
            result = await model.ainvoke("Call magic_function(3)")

    assert isinstance(result, AIMessage)
    assert not result.tool_calls
    assert len(result.invalid_tool_calls) == 1
    invalid_call = result.invalid_tool_calls[0]
    assert invalid_call["name"] == "magic_function"
    assert invalid_call["id"] == "call_test"
    assert invalid_call["args"] == arguments


@pytest.mark.parametrize("finish", ["separate", "same", "eof"])
@pytest.mark.parametrize(
    "fragments",
    [("{'input': 3}",), ("{'input':", " 3}"), ('{"input": 3}',), ('{"input":', " 3}")],
    ids=["malformed-whole", "malformed-split", "valid-whole", "valid-split"],
)
def test_stream_completes_tool_arguments(
    fragments: tuple[str, ...], finish: str
) -> None:
    transport = httpx.MockTransport(
        lambda request: _tool_response(
            request, malformed=False, multiple=False, fragments=fragments, finish=finish
        )
    )
    with httpx.Client(transport=transport) as client:
        model = ChatQwQ(api_key=SecretStr("test-key"), http_client=client)
        full = AIMessageChunk(content="")
        for chunk in model.stream("Call magic_function(3)"):
            full += chunk

    assert len(full.tool_calls) == 1
    assert full.tool_calls[0]["args"] == {"input": 3}
    assert full.tool_calls[0]["id"] == "call_test"
    assert len(full.tool_call_chunks) == 1
    assert json.loads(full.tool_call_chunks[0]["args"] or "") == {"input": 3}
    assert not full.invalid_tool_calls


@pytest.mark.parametrize("finish", ["separate", "same", "eof"])
@pytest.mark.parametrize(
    "fragments",
    [("{'input': 3}",), ("{'input':", " 3}"), ('{"input": 3}',), ('{"input":', " 3}")],
    ids=["malformed-whole", "malformed-split", "valid-whole", "valid-split"],
)
async def test_async_stream_completes_tool_arguments(
    fragments: tuple[str, ...], finish: str
) -> None:
    transport = httpx.MockTransport(
        lambda request: _tool_response(
            request, malformed=False, multiple=False, fragments=fragments, finish=finish
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        model = ChatQwQ(api_key=SecretStr("test-key"), http_async_client=client)
        full = AIMessageChunk(content="")
        async for chunk in model.astream("Call magic_function(3)"):
            full += chunk

    assert len(full.tool_calls) == 1
    assert full.tool_calls[0]["args"] == {"input": 3}
    assert full.tool_calls[0]["id"] == "call_test"
    assert len(full.tool_call_chunks) == 1
    assert json.loads(full.tool_call_chunks[0]["args"] or "") == {"input": 3}
    assert not full.invalid_tool_calls


@pytest.mark.parametrize("multiple", [False, True], ids=["single", "multiple"])
@pytest.mark.parametrize(
    "stream, malformed",
    [(False, False), (True, False), (False, True)],
    ids=["invoke", "stream", "invoke-malformed"],
)
async def test_async_tool_argument_fragments(
    stream: bool, malformed: bool, multiple: bool
) -> None:
    result: BaseMessage
    transport = httpx.MockTransport(
        lambda request: _tool_response(request, malformed=malformed, multiple=multiple)
    )
    async with httpx.AsyncClient(transport=transport) as client:
        model = ChatQwQ(api_key=SecretStr("test-key"), http_async_client=client)
        if stream:
            full = AIMessageChunk(content="")
            async for chunk in model.astream("Call magic_function(3)"):
                full += chunk
            result = full
        else:
            result = await model.ainvoke("Call magic_function(3)")

    assert isinstance(result, AIMessage)
    assert sorted(result.tool_calls, key=lambda call: call["name"]) == [
        {
            "name": "magic_function",
            "args": {"input": 3},
            "id": "call_test",
            "type": "tool_call",
        }
    ] + (
        [
            {
                "name": "other_function",
                "args": {"input": 4},
                "id": "call_second",
                "type": "tool_call",
            }
        ]
        if multiple
        else []
    )
    assert not result.invalid_tool_calls


def _assert_completed_result(message: BaseMessage, recorder: _Recorder) -> None:
    assert isinstance(message, AIMessage)
    assert message.id
    assert message.content == "Answer."
    assert message.additional_kwargs["reasoning_content"] == "Thinking."
    assert message.usage_metadata == {
        "input_tokens": 2,
        "output_tokens": 3,
        "total_tokens": 5,
        "input_token_details": {},
        "output_token_details": {},
    }
    assert message.response_metadata["finish_reason"] == "tool_calls"
    assert message.response_metadata["model_name"] == "test-model"
    assert message.response_metadata["model_provider"] == "dashscope"
    assert [call["args"] for call in message.tool_calls] == [{"input": 3}, {"input": 4}]
    assert [call["id"] for call in message.tool_calls] == ["call_test", "call_second"]
    assert not message.invalid_tool_calls
    raw_calls = message.additional_kwargs["tool_calls"]
    assert [json.loads(call["function"]["arguments"]) for call in raw_calls] == [
        {"input": 3},
        {"input": 4},
    ]
    assert [call["id"] for call in raw_calls] == ["call_test", "call_second"]

    # Other stream data arrives before completion; each completed tool emits once.
    assert recorder.chunks[0].content == "Answer."
    assert recorder.chunks[0].additional_kwargs["reasoning_content"] == "Thinking."
    assert not recorder.chunks[0].tool_call_chunks
    assert "tool_calls" not in recorder.chunks[0].additional_kwargs
    assert sum(len(chunk.tool_call_chunks) for chunk in recorder.chunks) == 2
    callback_full = AIMessageChunk(content="")
    for chunk in recorder.chunks:
        callback_full += chunk
    assert callback_full.tool_calls == message.tool_calls
    assert callback_full.additional_kwargs == message.additional_kwargs
    assert callback_full.content == message.content
    assert callback_full.usage_metadata == message.usage_metadata
    assert recorder.result is not None
    generation = recorder.result.generations[0][0]
    assert isinstance(generation, ChatGeneration)
    assert isinstance(generation.message, AIMessage)
    assert generation.message.tool_calls == message.tool_calls
    assert generation.message.additional_kwargs == message.additional_kwargs
    assert generation.message.usage_metadata == message.usage_metadata


@pytest.mark.parametrize("mode", ["stream", "invoke", "explicit-nonstream"])
@pytest.mark.parametrize("finish", ["separate", "same"])
def test_tool_completion_preserves_metadata_and_callbacks(
    mode: str, finish: str
) -> None:
    recorder = _Recorder()
    transport = httpx.MockTransport(
        lambda request: _tool_response(
            request, malformed=True, multiple=True, finish=finish
        )
    )
    result: BaseMessage
    with httpx.Client(transport=transport) as client:
        model = ChatQwQ(api_key=SecretStr("test-key"), http_client=client)
        if mode == "stream":
            full = AIMessageChunk(content="")
            for chunk in model.stream(
                "Call the tools", config={"callbacks": [recorder]}
            ):
                full += chunk
            result = full
        else:
            options: dict[str, Any] = (
                {"stream": False} if mode == "explicit-nonstream" else {}
            )
            result = model.invoke(
                "Call the tools", config={"callbacks": [recorder]}, **options
            )
    _assert_completed_result(result, recorder)


@pytest.mark.parametrize("mode", ["stream", "invoke", "explicit-nonstream"])
@pytest.mark.parametrize("finish", ["separate", "same"])
async def test_async_tool_completion_preserves_metadata_and_callbacks(
    mode: str, finish: str
) -> None:
    recorder = _Recorder()
    transport = httpx.MockTransport(
        lambda request: _tool_response(
            request, malformed=True, multiple=True, finish=finish
        )
    )
    result: BaseMessage
    async with httpx.AsyncClient(transport=transport) as client:
        model = ChatQwQ(api_key=SecretStr("test-key"), http_async_client=client)
        if mode == "stream":
            full = AIMessageChunk(content="")
            async for chunk in model.astream(
                "Call the tools", config={"callbacks": [recorder]}
            ):
                full += chunk
            result = full
        else:
            options: dict[str, Any] = (
                {"stream": False} if mode == "explicit-nonstream" else {}
            )
            result = await model.ainvoke(
                "Call the tools", config={"callbacks": [recorder]}, **options
            )
    _assert_completed_result(result, recorder)


def test_interrupted_stream_does_not_emit_partial_tool_calls() -> None:
    recorder = _Recorder()
    transport = httpx.MockTransport(
        lambda request: _tool_response(
            request, malformed=False, multiple=True, interrupt=True
        )
    )
    with httpx.Client(transport=transport) as client:
        model = ChatQwQ(api_key=SecretStr("test-key"), http_client=client)
        with pytest.raises(httpx.ReadError, match="Interrupted tool-call stream"):
            list(model.stream("Call the tools", config={"callbacks": [recorder]}))
    assert recorder.chunks
    assert recorder.result is None
    assert not any(chunk.tool_call_chunks for chunk in recorder.chunks)
    assert not any("tool_calls" in chunk.additional_kwargs for chunk in recorder.chunks)


async def test_async_interrupted_stream_does_not_emit_partial_tool_calls() -> None:
    recorder = _Recorder()
    transport = httpx.MockTransport(
        lambda request: _tool_response(
            request, malformed=False, multiple=True, interrupt=True
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        model = ChatQwQ(api_key=SecretStr("test-key"), http_async_client=client)
        with pytest.raises(httpx.ReadError, match="Interrupted tool-call stream"):
            async for _ in model.astream(
                "Call the tools", config={"callbacks": [recorder]}
            ):
                pass
    assert recorder.chunks
    assert recorder.result is None
    assert not any(chunk.tool_call_chunks for chunk in recorder.chunks)
    assert not any("tool_calls" in chunk.additional_kwargs for chunk in recorder.chunks)


@pytest.mark.parametrize("async_api", [False, True])
async def test_failed_repair_preserves_other_tools_and_invalid_data(
    monkeypatch: pytest.MonkeyPatch, async_api: bool
) -> None:
    repair_attempts = []

    def failing_repair(arguments: str) -> None:
        repair_attempts.append(arguments)
        raise ValueError("Cannot repair")

    monkeypatch.setattr("langchain_qwq.chat_models.json_repair.loads", failing_repair)
    transport = httpx.MockTransport(
        lambda request: _tool_response(request, malformed=True, multiple=True)
    )
    if async_api:
        async with httpx.AsyncClient(transport=transport) as async_client:
            model = ChatQwQ(
                api_key=SecretStr("test-key"), http_async_client=async_client
            )
            result = await model.ainvoke("Call the tools")
    else:
        with httpx.Client(transport=transport) as sync_client:
            model = ChatQwQ(api_key=SecretStr("test-key"), http_client=sync_client)
            result = model.invoke("Call the tools")

    assert repair_attempts == ["{'input': 3}"]
    assert result.tool_calls == [
        {
            "name": "other_function",
            "args": {"input": 4},
            "id": "call_second",
            "type": "tool_call",
        }
    ]
    assert len(result.invalid_tool_calls) == 1
    assert result.invalid_tool_calls[0]["args"] == "{'input': 3}"
    assert result.invalid_tool_calls[0]["id"] == "call_test"
