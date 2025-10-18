"""Test ChatQwQ chat model."""

from typing import Any, Type

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_tests.integration_tests import ChatModelIntegrationTests
import pytest

from langchain_qwq.chat_models import ChatQwen, ChatQwQ

load_dotenv()


class BaseChatQwenIntegrationTests(ChatModelIntegrationTests):
    @pytest.fixture
    def model(self, request: Any) -> BaseChatModel:
        """Model fixture."""
        extra_init_params = getattr(request, "param", None) or {}
        if extra_init_params.get("output_version") == "v1":
            pytest.skip("Output version v1 is not supported")
        return self.chat_model_class(
            **{
                **self.standard_chat_model_params,
                **self.chat_model_params,
                **extra_init_params,
            },
        )

    @property
    def has_tool_choice(self) -> bool:
        return False

    @property
    def supports_image_tool_message(self) -> bool:
        return False


class TestChatQwQIntegration(BaseChatQwenIntegrationTests):
    @pytest.fixture
    def model(self, request: Any) -> BaseChatModel:
        """Model fixture."""
        extra_init_params = getattr(request, "param", None) or {}
        if extra_init_params.get("output_version") == "v1":
            pytest.skip("Output version v1 is not supported")
        return self.chat_model_class(
            **{
                **self.standard_chat_model_params,
                **self.chat_model_params,
                **extra_init_params,
            },
        )

    @property
    def chat_model_class(self) -> Type[ChatQwQ]:
        return ChatQwQ

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "qwq-plus",
        }

    @pytest.mark.xfail(reason="Custom implementation for QwQ model")
    def test_stream(self, model: BaseChatModel) -> None:
        num_chunks = 0
        full: AIMessageChunk | None = None
        for chunk in model.stream("Hello"):
            assert chunk is not None
            assert isinstance(chunk, AIMessageChunk)
            assert isinstance(chunk.content, str | list)
            num_chunks += 1
            full = chunk if full is None else full + chunk
        assert num_chunks > 0
        assert isinstance(full, AIMessageChunk)
        assert full.content
        assert len(full.content_blocks) == 2
        # Qwen QwQ models are forced to think, so content_blocks are 2, and the first one is reasoning_content.
        assert full.content_blocks[0]["type"] == "reasoning"
        assert full.content_blocks[1]["type"] == "text"

    @pytest.mark.xfail(reason="Custom implementation for QwQ model")
    async def test_astream(self, model: BaseChatModel) -> None:
        num_chunks = 0
        full: AIMessageChunk | None = None
        async for chunk in model.astream("Hello"):
            assert chunk is not None
            assert isinstance(chunk, AIMessageChunk)
            assert isinstance(chunk.content, str | list)
            num_chunks += 1
            full = chunk if full is None else full + chunk
        assert num_chunks > 0
        assert isinstance(full, AIMessageChunk)
        assert full.content
        assert len(full.content_blocks) == 2
        # Qwen QwQ models are forced to think, so content_blocks are 2, and the first one is reasoning_content.
        assert full.content_blocks[0]["type"] == "reasoning"
        assert full.content_blocks[1]["type"] == "text"


class TestChatQwenIntegration(BaseChatQwenIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatQwen]:
        return ChatQwen

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {"model": "qwen-plus-latest"}

    @property
    def supports_json_mode(self) -> bool:
        """(bool) whether the chat model supports JSON mode."""
        return True
