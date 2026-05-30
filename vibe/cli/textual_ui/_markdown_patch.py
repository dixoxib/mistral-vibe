"""Monkey-patch Textual's MarkdownWidget.append() to offload
MarkdownIt.parse() to a thread pool executor.

The upstream append() calls parser.parse() synchronously, which
blocks the asyncio event loop for ~1-5ms per call. During streaming
LLM responses, this accumulates to noticeable stuttering.

We replace a single line: the sync parser.parse() call is wrapped
in run_in_executor(), matching what MarkdownWidget.update() already does.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_ref: dict[str, object] = {}


def _install() -> None:
    from textual.widgets._markdown import Markdown

    if "original" in _ref:
        return

    _ref["original"] = Markdown.append

    def patched_append(self: Markdown, markdown: str):
        from markdown_it import MarkdownIt as _MarkdownIt
        from textual.widgets._markdown import AwaitComplete, MarkdownHeader
        from textual.widgets.markdown import MarkdownBlock

        parser = (
            _MarkdownIt("gfm-like")
            if self._parser_factory is None
            else self._parser_factory()
        )

        self._markdown = self.source + markdown
        updated_source = "".join(
            self._markdown.splitlines(keepends=True)[self._last_parsed_line :]
        )

        async def await_append() -> None:
            async with self.lock:
                tokens = await asyncio.get_running_loop().run_in_executor(
                    None, parser.parse, updated_source
                )
                existing_blocks = [
                    child
                    for child in self.children
                    if isinstance(child, MarkdownBlock)
                ]
                start_line = self._last_parsed_line
                for token in reversed(tokens):
                    if token.map is not None and token.level == 0:
                        self._last_parsed_line += token.map[0]
                        break

                new_blocks = list(self._parse_markdown(tokens))
                any_headers = any(
                    isinstance(block, MarkdownHeader) for block in new_blocks
                )
                for block in new_blocks:
                    start, end = block.source_range
                    block.source_range = (start + start_line, end + start_line)

                with self.app.batch_update():
                    if existing_blocks and new_blocks:
                        last_block = existing_blocks[-1]
                        last_block.source_range = new_blocks[0].source_range
                        try:
                            await last_block._update_from_block(new_blocks[0])
                        except IndexError:
                            pass
                        else:
                            new_blocks = new_blocks[1:]

                    if new_blocks:
                        await self.mount_all(new_blocks)

                if any_headers:
                    self._table_of_contents = None
                    self.post_message(
                        Markdown.TableOfContentsUpdated(
                            self, self.table_of_contents
                        ).set_sender(self)
                    )

        return AwaitComplete(await_append())

    Markdown.append = patched_append
