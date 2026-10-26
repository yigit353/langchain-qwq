"""Tool arguments must survive JSON boundaries in provider stream deltas."""

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from pydantic import SecretStr

from langchain_qwq import ChatQwQ


def _tool_response(
    request: httpx.Request,
    *,
    malformed: bool,
    multiple: bool,
    fragments: tuple[str, str] = ('{"input":', " 3}"),
) -> httpx.Response:
    deltas: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_test",
                    "type": "function",
                    "function": {
                        "name": "magic_function",
                        "arguments": fragments[0],
                    },
                }
            ],
        },
        {"tool_calls": [{"index": 0, "function": {"arguments": fragments[1]}}]},
        {},
    ]
    if malformed:
        deltas[0]["tool_calls"][0]["function"]["arguments"] = "{'input':"
    if multiple:
        deltas[0]["tool_calls"].append(
            {
                "index": 1,
                "id": "call_second",
                "type": "function",
                "function": {"name": "other_function", "arguments": '{"input":'},
            }
        )
        deltas[1]["tool_calls"].append({"index": 1, "function": {"arguments": " 4}"}})
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
                    "finish_reason": "tool_calls" if not delta else None,
                }
            ],
        }
        for delta in deltas
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
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


@pytest.mark.parametrize("arguments", ["[]", "null"])
def test_non_object_tool_arguments_remain_invalid(arguments: str) -> None:
    transport = httpx.MockTransport(
        lambda request: _tool_response(
            request, malformed=False, multiple=False, fragments=(arguments, "")
        )
    )
    with httpx.Client(transport=transport) as client:
        model = ChatQwQ(api_key=SecretStr("test-key"), http_client=client)
        result = model.invoke("Call magic_function(3)")

    assert isinstance(result, AIMessage)
    assert not result.tool_calls
    assert len(result.invalid_tool_calls) == 1
    invalid_call = result.invalid_tool_calls[0]
    assert invalid_call["name"] == "magic_function"
    assert invalid_call["id"] == "call_test"
    assert invalid_call["args"] == arguments


@pytest.mark.parametrize("arguments", ["[]", "null"])
async def test_async_non_object_tool_arguments_remain_invalid(arguments: str) -> None:
    transport = httpx.MockTransport(
        lambda request: _tool_response(
            request, malformed=False, multiple=False, fragments=(arguments, "")
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        model = ChatQwQ(api_key=SecretStr("test-key"), http_async_client=client)
        result = await model.ainvoke("Call magic_function(3)")

    assert isinstance(result, AIMessage)
    assert not result.tool_calls
    assert len(result.invalid_tool_calls) == 1
    invalid_call = result.invalid_tool_calls[0]
    assert invalid_call["name"] == "magic_function"
    assert invalid_call["id"] == "call_test"
    assert invalid_call["args"] == arguments


def test_malformed_stream_preserves_raw_tool_arguments() -> None:
    transport = httpx.MockTransport(
        lambda request: _tool_response(request, malformed=True, multiple=False)
    )
    with httpx.Client(transport=transport) as client:
        model = ChatQwQ(api_key=SecretStr("test-key"), http_client=client)
        full = AIMessageChunk(content="")
        for chunk in model.stream("Call magic_function(3)"):
            full += chunk

    assert not full.tool_calls
    assert len(full.tool_call_chunks) == 1
    assert full.tool_call_chunks[0]["args"] == "{'input': 3}"
    assert len(full.invalid_tool_calls) == 1
    assert full.invalid_tool_calls[0]["args"] == "{'input': 3}"
    assert full.invalid_tool_calls[0]["id"] == "call_test"


async def test_async_malformed_stream_preserves_raw_tool_arguments() -> None:
    transport = httpx.MockTransport(
        lambda request: _tool_response(request, malformed=True, multiple=False)
    )
    async with httpx.AsyncClient(transport=transport) as client:
        model = ChatQwQ(api_key=SecretStr("test-key"), http_async_client=client)
        full = AIMessageChunk(content="")
        async for chunk in model.astream("Call magic_function(3)"):
            full += chunk

    assert not full.tool_calls
    assert len(full.tool_call_chunks) == 1
    assert full.tool_call_chunks[0]["args"] == "{'input': 3}"
    assert len(full.invalid_tool_calls) == 1
    assert full.invalid_tool_calls[0]["args"] == "{'input': 3}"
    assert full.invalid_tool_calls[0]["id"] == "call_test"


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
