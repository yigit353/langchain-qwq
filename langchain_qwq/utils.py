from typing import AsyncIterator, Iterator, Tuple

from langchain_core.messages import AIMessageChunk, BaseMessageChunk


def convert_reasoning_to_content(
    model_response: Iterator[BaseMessageChunk],
    think_tag: Tuple[str, str] = ("<think>", "</think>"),
) -> Iterator[BaseMessageChunk]:
    isfirst = True
    isend = True

    for chunk in model_response:
        if (
            isinstance(chunk, AIMessageChunk)
            and "reasoning_content" in chunk.additional_kwargs
        ):
            if isfirst:
                chunk.content = (
                    f"{think_tag[0]}{chunk.additional_kwargs['reasoning_content']}"
                )
                isfirst = False
            else:
                chunk.content = chunk.additional_kwargs["reasoning_content"]
        elif (
            isinstance(chunk, AIMessageChunk)
            and "reasoning_content" not in chunk.additional_kwargs
            and chunk.content
            and isend
        ):
            chunk.content = f"{think_tag[1]}{chunk.content}"
            isend = False
        yield chunk


async def aconvert_reasoning_to_content(
    amodel_response: AsyncIterator[BaseMessageChunk],
    think_tag: Tuple[str, str] = ("<think>", "</think>"),
) -> AsyncIterator[BaseMessageChunk]:
    isfirst = True
    isend = True
    async for chunk in amodel_response:
        if (
            isinstance(chunk, AIMessageChunk)
            and "reasoning_content" in chunk.additional_kwargs
        ):
            if isfirst:
                chunk.content = (
                    f"{think_tag[0]}{chunk.additional_kwargs['reasoning_content']}"
                )
                isfirst = False
            else:
                chunk.content = chunk.additional_kwargs["reasoning_content"]
        elif (
            isinstance(chunk, AIMessageChunk)
            and "reasoning_content" not in chunk.additional_kwargs
            and chunk.content
            and isend
        ):
            chunk.content = f"{think_tag[1]}{chunk.content}"
            isend = False
        yield chunk
