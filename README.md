# langchain-qwq

This package provides seamless integration between LangChain and QwQ models as well as other Qwen series models from Alibaba Cloud BaiLian (via OpenAI-compatible API).

## Features

- **QwQ Model Integration**: Full support for QwQ models with advanced reasoning capabilities
- **Qwen3 Model Integration**: Comprehensive support for Qwen3 series models with hybrid reasoning modes
- **Other Qwen Models**: Compatibility with Qwen-Max, Qwen2.5, and other Qwen series models
- **Vision Models**: Native support for Qwen-VL series vision models
- **Streaming Support**: Synchronous and asynchronous streaming capabilities
- **Tool Calling**: Function calling with support for parallel execution
- **Structured Output**: JSON mode and function calling for structured response generation
- **Reasoning Access**: Direct access to internal model reasoning and thinking content

## Installation

To install the package:

```bash
pip install -U langchain-qwq
```

If you want to install additional dependencies for development:

```bash
pip install -U langchain-qwq[test]
pip install -U langchain-qwq[codespell]
pip install -U langchain-qwq[lint]
pip install -U langchain-qwq[typing]
```

**Note**: The documentation notebooks in `docs/` can be viewed directly on GitHub or in VS Code without additional dependencies. To run them interactively, install Jupyter separately: `pip install jupyterlab`

## Environment Variables

Authentication and configuration are managed through the following environment variables:

- `DASHSCOPE_API_KEY`: Your DashScope API key (required)
- `DASHSCOPE_API_BASE`: Optional API base URL (defaults to `"https://dashscope-intl.aliyuncs.com/compatible-mode/v1"`)

> **Note**: Domestic Chinese users should configure `DASHSCOPE_API_BASE` to the domestic endpoint, as `langchain-qwq` defaults to the international Alibaba Cloud endpoint.

## ChatQwQ

The ChatQwQ class provides access to QwQ chat models with built-in reasoning capabilities.

### Basic Usage

```python
from langchain_qwq import ChatQwQ

model = ChatQwQ(model="qwq-plus")
response = model.invoke("Hello, how are you?")
print(response.content)
```

### Accessing Reasoning Content

You can access the internal reasoning content of QwQ models via `additional_kwargs`:

```python
response = model.invoke("Hello")
content = response.content
reasoning = response.additional_kwargs.get("reasoning_content", "")
print(f"Response: {content}")
print(f"Reasoning: {reasoning}")
```

### Streaming

#### Sync Streaming

```python
model = ChatQwQ(model="qwq-plus")

is_first = True
is_end = True

for msg in model.stream("Hello"):
    if hasattr(msg, 'additional_kwargs') and "reasoning_content" in msg.additional_kwargs:
        if is_first:
            print("Starting to think...")
            is_first = False
        print(msg.additional_kwargs["reasoning_content"], end="", flush=True)
    elif hasattr(msg, 'content') and msg.content:
        if is_end:
            print("\nThinking ended")
            is_end = False
        print(msg.content, end="", flush=True)
```

#### Async Streaming

```python
is_first = True
is_end = True

async for msg in model.astream("Hello"):
    if hasattr(msg, 'additional_kwargs') and "reasoning_content" in msg.additional_kwargs:
        if is_first:
            print("Starting to think...")
            is_first = False
        print(msg.additional_kwargs["reasoning_content"], end="", flush=True)
    elif hasattr(msg, 'content') and msg.content:
        if is_end:
            print("\nThinking ended")
            is_end = False
        print(msg.content, end="", flush=True)
```

### Using Content Blocks

ChatQwQ also supports v1 version content_blocks, for example

```python
from langchain_qwq import ChatQwen, ChatQwQ
model = ChatQwQ(model="qwq-plus")
print(model.invoke("Hello").content_blocks)

```

### Tool Calling

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city"""
    return f"The weather in {city} is sunny."

bound_model = model.bind_tools([get_weather])
response = bound_model.invoke("What's the weather in New York?")
print(response.tool_calls)
```

### Structured Output

#### JSON Mode

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

struct_model = model.with_structured_output(User, method="json_mode")
response = struct_model.invoke("Hello, I'm John and I'm 25 years old")
print(response)  # User(name='John', age=25)
```

#### Function Calling Mode

```python
struct_model = model.with_structured_output(User, method="function_calling")
response = struct_model.invoke("My name is Alice and I'm 30")
print(response)  # User(name='Alice', age=30)
```

### Integration with LangChain Agents

```python
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_qwq import ChatQwen, ChatQwQ


@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."


model = ChatQwQ(model="qwq-plus")
agent = create_agent(model, tools=[get_weather])
print(agent.invoke({"messages": [HumanMessage("What is the weather like in New York?")]}))
```

### QvQ Model Example

```python
from langchain_core.messages import HumanMessage

messages = [
    HumanMessage(
        content_blocks=[
            {
                "type": "image",
                "url": "https://www.example.com/image.jpg",
            },
            {"type": "text", "text": "describe the image"},
        ]
    )
]


# model = ChatQwen(model="qwen-plus-latest")
model = ChatQwQ(model="qvq-plus")
print(model.invoke(messages))

```

## ChatQwen

The ChatQwen class offers enhanced support for Qwen3 and other Qwen series models, including specialized parameters for Qwen3's thinking mode.

### Basic Usage

```python
from langchain_qwq import ChatQwen

# Qwen3 model
model = ChatQwen(model="qwen3-235b-a22b-instruct-2507")
response = model.invoke("Hello")
print(response.content)

model=ChatQwen(model="qwen3-235b-a22b-thinking-2507")
response=model.invoke("Hello")
# Access reasoning content (Qwen3 only)
reasoning = response.additional_kwargs.get("reasoning_content", "")
print(f"Reasoning: {reasoning}")
```

### Thinking Control

> **Note**: This feature is only applicable to Qwen3 models. It applies to all Qwen3 models except the latest ones, including but not limited to `Qwen3-235b-a22b-thinking-2507`, `Qwen3-235b-a22b-instruct-2507`, `Qwen3-Coder-480B-a35b-instruct`, and `Qwen3-Coder-plus`.

#### Disable Thinking Mode

```python
# Disable thinking for open-source Qwen3 models
model = ChatQwen(model="qwen3-32b", enable_thinking=False)
response = model.invoke("Hello")
print(response.content)  # No reasoning content
```

#### Enable Thinking for Proprietary Models

```python
# Enable thinking for proprietary models
model = ChatQwen(model="qwen-plus-latest", enable_thinking=True)
response = model.invoke("Hello")
reasoning = response.additional_kwargs.get("reasoning_content", "")
print(f"Reasoning: {reasoning}")
```

#### Control Thinking Length

```python
# Set thinking budget (max thinking tokens)
model = ChatQwen(model="qwen3-32b", thinking_budget=20)
response = model.invoke("Hello")
reasoning = response.additional_kwargs.get("reasoning_content", "")
print(f"Limited reasoning: {reasoning}")
```

### Other Qwen Models

#### Qwen-Max

```python
model = ChatQwen(model="qwen-max-latest")
print(model.invoke("Hello").content)

# Tool calling
bound_model = model.bind_tools([get_weather])
response = bound_model.invoke("Weather in Shanghai and Beijing?", parallel_tool_calls=True)
print(response.tool_calls)

# Structured output
struct_model = model.with_structured_output(User, method="json_mode")
result = struct_model.invoke("I'm Bob, 28 years old")
print(result)
```

#### Qwen2.5-72B

```python
model = ChatQwen(model="qwen2.5-72b-instruct")
print(model.invoke("Hello").content)

# All features work the same as other models
bound_model = model.bind_tools([get_weather])
struct_model = model.with_structured_output(User, method="json_mode")
```

### Using Content Blocks

ChatQwen supports content blocks, for example

```python
from langchain_qwq import ChatQwen

model = ChatQwen(model="qwen-plus-latest",enable_thinking=True)
print(model.invoke("Hello").content_blocks)
```

### Integration with LangChain Agents

```python
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_qwq import ChatQwen


@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."


model = ChatQwen(model="qwen-plus-latest")
agent = create_agent(model, tools=[get_weather])
print(agent.invoke({"messages": [HumanMessage("查询New York的天气")]}))
```

### Vision Models

```python
from langchain_core.messages import HumanMessage
from langchain_qwq import ChatQwen


messages = [
    HumanMessage(
        content_blocks=[
            {
                "type": "image",
                "url": "https://www.example.com/image.jpg",
            },
            {"type": "text", "text": "描述图片内容"},
        ]
    )
]


model = ChatQwen(model="qwen3-vl-plus")
print(model.invoke(messages))
```

## Middleware

This library provides a middleware `DashScopeContextCacheMiddleware` that can be used to create display caching. 

Example usage

```python
from langchain.agents import create_agent
from langchain.tools import tool

from langchain_qwq import ChatQwen
from langchain_qwq.middleware import DashScopeContextCacheMiddleware


@tool
def get_weather(city: str) -> str:
    """Get the current weather in a given city."""
    return f"The weather in {city} is 20 degrees Celsius."


model = ChatQwen(model="qwen3-max")

agent = create_agent(
    model, tools=[get_weather], middleware=[DashScopeContextCacheMiddleware()]
)
```

## Model Comparison

| Feature           | ChatQwQ    | ChatQwen        |
| ----------------- | ---------- | --------------- |
| QwQ Models        | ✅ Primary | ❌              |
| QvQ Models        | ✅ Primary | ❌              |
| Qwen3 Models      | ✅ Basic   | ✅ Enhanced     |
| Other Qwen Models | ❌         | ✅ Full Support |
| Vision Models     | ❌         | ✅ Supported    |
| Thinking Control  | ❌         | ✅ (Qwen3 only) |
| Thinking Budget   | ❌         | ✅ (Qwen3 only) |

### Usage Guidance

- Use ChatQwQ for QwQ and QvQ models.
- For Qwen3 series models (available only on Alibaba Cloud BAILIAN platform) with deep thinking mode enabled, all invocations will automatically use streaming.
- For other Qwen series models (including self-deployed or third-party deployed Qwen3 models), use ChatQwen, and streaming will not be automatically enabled.
