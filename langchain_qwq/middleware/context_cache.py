from typing import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ModelCallResult


class DashScopeContextCacheMiddleware(AgentMiddleware):
    """Middleware for caching context in DashScope API.

    Please refer to https://help.aliyun.com/zh/model-studio/context-cache
    for more details.

    Example:
        ```python
        from langchain_qwq.middleware import DashScopeContextCacheMiddleware
        ```
    """

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelCallResult:
        messages = request.state.get("messages", [])
        message = messages[-1]

        request = request.override(
            messages=[
                *messages[:-1],
                message.model_copy(
                    update={
                        "content": [
                            {
                                "type": "text",
                                "text": message.content,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ]
                    }
                ),
            ]
        )
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        messages = request.state.get("messages", [])
        message = messages[-1]

        request = request.override(
            messages=[
                *messages[:-1],
                message.model_copy(
                    update={
                        "content": [
                            {
                                "type": "text",
                                "text": message.content,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ]
                    }
                ),
            ]
        )
        return await handler(request)
