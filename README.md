# langchain-qwq

LangChain chat models for Alibaba Cloud Model Studio (DashScope) and compatible
Qwen endpoints. The package provides `ChatQwQ` and `ChatQwen`, with synchronous and
asynchronous calls, streaming, tool calls, structured output, and access to
provider-returned reasoning content. Available features depend on the model and
endpoint you select.

## Upgrading to 0.4.0

Alibaba retired `qwq-plus` in the Beijing and Singapore regions on **October 10,
2026**. Calls to that model can now return HTTP 404 `model_not_found`. See the
[retirement notice](https://www.alibabacloud.com/en/notice/model_studionotice_on_the_decommissioning_of_certain_historical_mainline_models_731).

The package name and imports are unchanged. To keep using `ChatQwQ`, explicitly
select a supported model and enable its thinking mode:

```python
from langchain_qwq import ChatQwQ

model = ChatQwQ(
    model="qwen-plus-latest",
    extra_body={"enable_thinking": True},
)
```

This configuration is covered by the live integration tests. **The `ChatQwQ`
constructor still defaults to `qwq-plus`**, so do not omit `model` when using
DashScope. The release does not silently redirect requests to a different model.
For first-class `enable_thinking` and `thinking_budget` arguments, use `ChatQwen`
as shown below. Check model availability in your account and region; a model
identifier supported by one provider may not be supported by another.

Version 0.4.0 also fixes fragmented tool-call arguments in `ChatQwQ`. Tool-call
arguments are buffered until completion, then valid JSON is preserved and
repairable malformed JSON is repaired. Text and reasoning continue to stream
without waiting for the tool call. See [CHANGELOG.md](https://github.com/yigit353/langchain-qwq/blob/codex/release-0.4.0/CHANGELOG.md)
for the release details.

## Installation and configuration

Requires Python 3.10 or newer.

```bash
pip install -U langchain-qwq
```

Set `DASHSCOPE_API_KEY` to your real DashScope API key in your environment before
running the examples. `DASHSCOPE_API_BASE` optionally overrides the base URL;
the package default is:

```text
https://dashscope-intl.aliyuncs.com/compatible-mode/v1
```

Use the endpoint and credentials for your region. For a different provider or
self-hosted server, pass its `base_url` and `api_key` explicitly. Model access,
thinking controls, tool calling, and image inputs must be supported by that
endpoint.

## Quickstart

This is a complete synchronous example after configuring your API key:

```python
from langchain_qwq import ChatQwQ

model = ChatQwQ(
    model="qwen-plus-latest",
    extra_body={"enable_thinking": True},
)
response = model.invoke("What is 17 multiplied by 19?")
print(response.content)
print(response.additional_kwargs.get("reasoning_content", ""))
```

Alternatively, use `ChatQwen` to configure thinking directly:

```python
from langchain_qwq import ChatQwen

model = ChatQwen(model="qwen-plus-latest", enable_thinking=True)
response = model.invoke("What is 17 multiplied by 19?")
print(response.content)
```

### Choosing a wrapper

| Behavior | `ChatQwQ` | `ChatQwen` |
| --- | --- | --- |
| Model selection | Explicitly select an active model; default remains retired `qwq-plus` | Explicitly select a model; default is `qwen3-32b` |
| Thinking controls | Provider parameters through `extra_body` | `enable_thinking`, `thinking_budget`, and `extra_body` |
| Transport for `invoke` / `ainvoke` | Always streams internally and assembles the response | Automatically enables streaming for detected thinking models on DashScope endpoints |
| Tool-argument streaming | Buffers and repairs complete arguments | Uses the underlying streaming behavior |
| Structured output | Function calling; other method names are converted to function calling | Function calling or JSON mode, subject to model support |
| Reasoning content | `additional_kwargs["reasoning_content"]` when returned | Same; also maps supported OpenRouter reasoning responses |

Neither wrapper makes every feature available on every Qwen model. Use a model
with the capabilities your application needs.

## Reasoning and content blocks

The answer is in `response.content`. When the provider returns reasoning, it is
available separately:

```python
reasoning = response.additional_kwargs.get("reasoning_content", "")
print(reasoning)
print(response.content)
print(response.content_blocks)
```

`content_blocks` is LangChain's normalized view of the response and can include
reasoning and text blocks. Accessing that property does **not** mean this package
supports `output_version="v1"`; that output mode is not supported here. Reasoning
is present only when the selected model and request produce it.

## Streaming and async calls

The following continues with `model` from the `ChatQwQ` quickstart:

```python
for chunk in model.stream("What is 17 multiplied by 19?"):
    reasoning = chunk.additional_kwargs.get("reasoning_content", "")
    if reasoning:
        print(reasoning, end="", flush=True)
    if chunk.content:
        print(chunk.content, end="", flush=True)
print()
```

A complete asynchronous example:

```python
import asyncio

from langchain_qwq import ChatQwQ


async def main() -> None:
    model = ChatQwQ(
        model="qwen-plus-latest",
        extra_body={"enable_thinking": True},
    )
    response = await model.ainvoke("What is 17 multiplied by 19?")
    print(response.content)

    async for chunk in model.astream("Explain the calculation briefly."):
        reasoning = chunk.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            print(reasoning, end="", flush=True)
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


asyncio.run(main())
```

In a notebook with an active event loop, use `await main()` instead of
`asyncio.run(main())`.

## Tool calling and agents

Bind tools to a model that supports function calling:

```python
from langchain_core.tools import tool
from langchain_qwq import ChatQwQ


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


model = ChatQwQ(
    model="qwen-plus-latest",
    extra_body={"enable_thinking": True},
)
response = model.bind_tools([multiply]).invoke("Use multiply to calculate 17 * 19.")
print(response.tool_calls)
```

`bind_tools` supplies tool definitions and returns requested tool calls; it does
not execute those tools. An agent can handle the tool execution loop:

```python
from langchain.agents import create_agent

agent = create_agent(model, tools=[multiply])
result = agent.invoke({"messages": [("user", "Use multiply to calculate 17 * 19.")]})
print(result["messages"][-1].content)
```

For `ChatQwQ` streaming, completed tool calls appear when the provider finishes
the generation, or at a clean end of stream if it omits a finish marker. Do not
expect executable argument fragments in earlier chunks. Provider errors and
cancelled streams do not flush unfinished calls. This buffering applies to
synchronous and asynchronous streams, including streaming callbacks.

## Structured output

`ChatQwQ` uses function calling for structured output. Passing `"json_mode"` or
`"json_schema"` to this wrapper also selects function calling; it does not enable
native JSON mode.

```python
from pydantic import BaseModel
from langchain_qwq import ChatQwQ


class Person(BaseModel):
    name: str
    age: int


model = ChatQwQ(
    model="qwen-plus-latest",
    extra_body={"enable_thinking": True},
)
structured = model.with_structured_output(Person, method="function_calling")
person = structured.invoke("Extract the person: Alice is 30 years old.")
print(person)
```

`ChatQwen` also supports `method="json_mode"` with a compatible model. Use a
separate instance with thinking disabled for this example:

```python
from langchain_qwq import ChatQwen

json_model = ChatQwen(model="qwen-plus-latest", enable_thinking=False)
structured = json_model.with_structured_output(Person, method="json_mode")
print(structured.invoke("Extract the person: Alice is 30 years old."))
```

For `ChatQwen`, structured-output setup can disable its `enable_thinking` flag
for hybrid models. Models whose names contain `thinking` use function calling
instead. `"json_schema"` is mapped to JSON mode by the base implementation; it
is not a separate strict-schema API. A Pydantic schema validates the parsed
result. Use `include_raw=True` to receive `raw`, `parsed`, and `parsing_error`
fields when inspecting parsing failures.

## ChatQwen thinking controls

Use controls supported by the selected model. On DashScope, for example:

```python
from langchain_qwq import ChatQwen

model = ChatQwen(
    model="qwen-plus-latest",
    enable_thinking=True,
    thinking_budget=100,
)
print(model.invoke("What is 17 multiplied by 19?").content)
```

`thinking_budget` limits thinking tokens where the model supports it. It does
not guarantee a particular answer length. See Alibaba's
[thinking-mode documentation](https://www.alibabacloud.com/help/en/model-studio/deep-thinking)
for current model support and limits.

Some OpenAI-compatible servers expect thinking control inside
`chat_template_kwargs`:

```python
from langchain_qwq import ChatQwen

model = ChatQwen(
    model="Qwen/Qwen3.6-35B-A3B",
    base_url="http://localhost:8000/v1",
    api_key="test-key",  # Placeholder for a local server without authentication.
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

Omit the separate `enable_thinking` constructor argument in this form. The nested
value takes precedence if both forms are present. For structured output, select
a method supported by the server, such as `method="json_mode"` where available.
This local example assumes you have already served the named model.

## Vision inputs

Use `ChatQwen` with an image-capable model available at your endpoint. A typical
message has this shape; replace `image_url` with an accessible image URL:

```python
from langchain_core.messages import HumanMessage
from langchain_qwq import ChatQwen

vision_model = ChatQwen(model="qwen3-vl-plus")
message = HumanMessage(
    content_blocks=[
        {"type": "image", "url": image_url},
        {"type": "text", "text": "Describe this image."},
    ]
)
response = vision_model.invoke([message])
```

Text-only models cannot gain vision support from this message format. Confirm the
model identifier and image support in your provider's current model catalog.

## DashScope context-cache middleware

`DashScopeContextCacheMiddleware` adds an explicit cache marker to the latest
message. Use it with a DashScope model that supports that cache format and with
text content suitable for the middleware:

```python
from langchain.agents import create_agent
from langchain_qwq import ChatQwen
from langchain_qwq.middleware import DashScopeContextCacheMiddleware

model = ChatQwen(model="qwen-plus-latest")
agent = create_agent(model, middleware=[DashScopeContextCacheMiddleware()])
```

Cache availability and billing depend on the selected model and endpoint; see
[DashScope context caching](https://help.aliyun.com/zh/model-studio/context-cache).
The middleware does not provide a local response cache.

## Development

From a clone of this repository, install the Poetry dependency groups:

```bash
poetry install --with test,lint,typing,codespell,dev
make format
make lint
make test
make spell_check
make check_imports
make integration_test
```

Unit tests disable network sockets. Integration tests use real provider APIs and
require a valid `DASHSCOPE_API_KEY`; local test credentials can be placed in
`tests/.env`. The suite tests `ChatQwQ` with `qwen-plus-latest` in thinking mode and
`ChatQwen` with `qwen-plus-latest`. Model availability and API charges apply.

The notebooks in `docs/` can be viewed on GitHub or in VS Code. Install Jupyter
separately if you want to run them interactively. Older notebooks may use retired
model identifiers; apply the migration above before running them.
