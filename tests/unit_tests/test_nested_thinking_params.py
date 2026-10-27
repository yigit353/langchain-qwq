"""Request-shape tests for ChatQwen thinking parameters."""

import json
from typing import Any, Literal

import httpx
import pytest
from pydantic import BaseModel

from langchain_qwq import ChatQwen


class Answer(BaseModel):
    answer: str


StructuredMethod = Literal["default", "function_calling", "json_mode", "json_schema"]


def _completion(
    request: httpx.Request, payloads: list[dict[str, Any]]
) -> httpx.Response:
    payload = json.loads(request.content)
    payloads.append(payload)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": '{"answer":"ok"}',
    }
    if "tools" in payload:
        message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-test",
                    "type": "function",
                    "function": {
                        "name": "Answer",
                        "arguments": '{"answer":"ok"}',
                    },
                }
            ],
        }
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        },
        request=request,
    )


async def _capture_request(
    *,
    use_async: bool,
    structured_method: StructuredMethod | None = None,
    **model_kwargs: Any,
) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    transport = httpx.MockTransport(lambda request: _completion(request, payloads))

    if use_async:
        async with httpx.AsyncClient(transport=transport) as client:
            model = ChatQwen(http_async_client=client, **model_kwargs)
            if structured_method == "default":
                runnable = model.with_structured_output(Answer)
            elif structured_method:
                runnable = model.with_structured_output(
                    Answer, method=structured_method
                )
            else:
                runnable = model
            await runnable.ainvoke("Return an answer")
    else:
        with httpx.Client(transport=transport) as client:
            model = ChatQwen(http_client=client, **model_kwargs)
            if structured_method == "default":
                runnable = model.with_structured_output(Answer)
            elif structured_method:
                runnable = model.with_structured_output(
                    Answer, method=structured_method
                )
            else:
                runnable = model
            runnable.invoke("Return an answer")

    assert len(payloads) == 1
    return payloads[0]


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("nested_value", [False, True])
async def test_nested_thinking_takes_precedence_for_ordinary_requests(
    use_async: bool, nested_value: bool
) -> None:
    payload = await _capture_request(
        use_async=use_async,
        model="qwen3-32b",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        enable_thinking=not nested_value,
        extra_body={
            "request_setting": "kept",
            "chat_template_kwargs": {
                "enable_thinking": nested_value,
                "template_setting": "kept",
            },
        },
    )

    assert "enable_thinking" not in payload
    assert payload["request_setting"] == "kept"
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": nested_value,
        "template_setting": "kept",
    }


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    ("model_name", "nested_value", "structured_method"),
    [
        ("Qwen/Qwen3.6-35B-A3B", False, "default"),
        ("qwen3.6-35b-a3b", True, "json_mode"),
    ],
)
async def test_nested_thinking_takes_precedence_for_structured_requests(
    use_async: bool,
    model_name: str,
    nested_value: bool,
    structured_method: StructuredMethod,
) -> None:
    payload = await _capture_request(
        use_async=use_async,
        structured_method=structured_method,
        model=model_name,
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        extra_body={
            "request_setting": "kept",
            "chat_template_kwargs": {
                "enable_thinking": nested_value,
                "template_setting": "kept",
            },
        },
    )

    assert "enable_thinking" not in payload
    assert payload["request_setting"] == "kept"
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": nested_value,
        "template_setting": "kept",
    }


@pytest.mark.parametrize(
    ("enable_thinking", "extra_body", "expected"),
    [
        (True, None, {"enable_thinking": True}),
        (
            False,
            {
                "request_setting": "kept",
                "chat_template_kwargs": {"template_setting": "kept"},
            },
            {
                "enable_thinking": False,
                "request_setting": "kept",
                "chat_template_kwargs": {"template_setting": "kept"},
            },
        ),
        (
            True,
            {"chat_template_kwargs": ["kept"]},
            {"enable_thinking": True, "chat_template_kwargs": ["kept"]},
        ),
        (
            True,
            {
                "enable_thinking": "caller-value",
                "chat_template_kwargs": {"enable_thinking": False},
            },
            {
                "enable_thinking": "caller-value",
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
    ],
)
async def test_existing_top_level_thinking_behavior_is_preserved(
    enable_thinking: bool,
    extra_body: dict[str, Any] | None,
    expected: dict[str, Any],
) -> None:
    payload = await _capture_request(
        use_async=False,
        model="qwen3-32b",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        enable_thinking=enable_thinking,
        extra_body=extra_body,
    )

    for key, value in expected.items():
        assert payload[key] == value
