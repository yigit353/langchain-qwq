# Changelog

## 0.4.0

- Require patched AnyIO for TLS hostname verification and a compatible LangGraph dependency set.
- Preserve valid streamed tool arguments and repair malformed JSON only after each complete tool call is assembled. Text and reasoning continue streaming immediately; unrepaired arguments remain inspectable.
- Keep synchronous and asynchronous invocation, streaming, and callbacks consistent, including multiple tool calls and interrupted streams.
- Respect explicit nested thinking configuration in `ChatQwen`.
- Restore Python 3.10 compatibility through the `typing-extensions` backport of `Self`.
- Update examples for supported Qwen thinking models following the retirement of DashScope's `qwq-plus` endpoint. `ChatQwQ` remains available for compatible APIs; its public default model is unchanged.
