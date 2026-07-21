"""Langfuse render adapter — export-time presentation transform for the Langfuse target.

The tracing pipeline records faithful OpenInference/LangChain shapes. Langfuse displays an
observation's input/output from the ``input.value``/``output.value`` attributes and pretty-renders
them only when they hold OpenAI-format chat messages (role/content, tool calls in ``tool_calls``
with JSON-string arguments). LangChain's serialized forms — ``{type, data}`` dicts, lc-constructor
dicts, ``Command``/update-chunk envelopes, raw ``LLMResult`` — all fall back to raw JSON in the UI.

This adapter rewrites those two attributes — and only those — on spans bound for Langfuse:

===================  =================================  ====================================
span class           input.value                        output.value
===================  =================================  ====================================
graph-root           messages -> OpenAI format          last message, converted (the answer)
``tools`` node       the triggering assistant           update-chunks -> tool messages
                     tool-call message
other nodes          DROPPED (dedup: full-state copy)   delta messages, converted
generation (LLM)     full message list, converted       LLMResult -> assistant message
tool                 kept (args are already clean)      Command/ToolMessage envelope -> content
===================  =================================  ====================================

Graph-root vs node uses the LangGraph invariant: a node run's name equals the
``langgraph_node`` it executes as (recorded in the span's ``metadata`` attribute); a Pregel
graph-root's name never does — including subagent roots, which inherit the *parent* node's
``langgraph_node`` through config propagation.

Reasoning: vLLM's ``reasoning_content`` (preserved by the serving layer in the message's
``additional_kwargs``) is re-attached to the converted assistant message as Langfuse's
registered ``thinking`` content field. ``convert_to_openai_messages`` alone would drop it.

Safety model:
- The dedup DROP is **fail-closed**: node-span inputs are removed before any conversion runs,
  so a conversion error can never re-inflate the trace (Langfuse ingest rejects >~4.5 MB).
- Conversions are **fail-open per value**: on any error the raw value passes through unchanged
  and the span is stamped with ``aegra.render.failed`` for the audit script to count.
- Spans are never mutated: a delegating wrapper overrides ``.attributes`` only, and the
  original span is exported untouched if rendering throws.
"""

import json
import logging
from collections.abc import Sequence
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)

INPUT_VALUE = "input.value"
INPUT_MIME = "input.mime_type"
OUTPUT_VALUE = "output.value"
OUTPUT_MIME = "output.mime_type"
SPAN_KIND = "openinference.span.kind"
METADATA = "metadata"
JSON_MIME = "application/json"
RENDER_FAILED = "aegra.render.failed"

_CHAIN_KINDS = ("CHAIN", "AGENT")


# ---------------------------------------------------------------------------
# message decoding: recorded forms -> LangChain messages -> OpenAI format
# ---------------------------------------------------------------------------


def _decode_messages(items: Sequence[Any]) -> list[Any]:
    """Turn a recorded message collection into LangChain ``BaseMessage`` objects.

    Three forms appear in recorded spans, sometimes mixed:
    - ``messages_to_dict`` form: ``{"type": "human", "data": {...}}``
    - lc-constructor form: ``{"lc": 1, "type": "constructor", "id": [...], "kwargs": {...}}``
    - already-OpenAI form: ``{"role": ..., "content": ...}`` (raw run inputs; passed through —
      ``convert_to_openai_messages`` accepts these natively)
    """
    from langchain_core.load import load
    from langchain_core.messages import BaseMessage, messages_from_dict

    decoded: list[Any] = []
    for item in items:
        if isinstance(item, BaseMessage):
            decoded.append(item)
        elif isinstance(item, dict) and item.get("lc") == 1:
            decoded.append(load(item, allowed_objects="messages"))
        elif isinstance(item, dict) and "data" in item and "type" in item:
            decoded.extend(messages_from_dict([item]))
        elif isinstance(item, dict) and "role" in item:
            decoded.append(item)
        else:
            raise ValueError(f"unrecognized message form: {list(item)[:5] if isinstance(item, dict) else type(item)}")
    return decoded


def _reasoning_of(message: Any) -> str | None:
    kwargs = getattr(message, "additional_kwargs", None) or {}
    reasoning = kwargs.get("reasoning_content") or kwargs.get("reasoning")
    return reasoning if isinstance(reasoning, str) and reasoning.strip() else None


def _to_openai(messages: Sequence[Any]) -> list[dict[str, Any]]:
    """Convert LangChain messages to OpenAI format, re-attaching reasoning.

    Converted one message at a time so reasoning stays associated with its own
    message. Reasoning goes into Langfuse's registered ``thinking`` field
    (``ThinkingContentPartSchema``) — the converter itself drops it.
    """
    from langchain_core.messages import convert_to_openai_messages

    out: list[dict[str, Any]] = []
    for message in messages:
        converted = convert_to_openai_messages([message])
        reasoning = _reasoning_of(message)
        if reasoning and converted and converted[0].get("role") == "assistant":
            converted[0]["thinking"] = [{"type": "thinking", "content": reasoning}]
        out.extend(converted)
    return out


def _messages_of(value: Any) -> list[Any] | None:
    """Extract the message list from a recorded I/O payload, if there is one.

    Handles ``{"messages": [...]}`` state dicts (also the generation form, where
    ``messages`` is a list of lists), and bare message lists.
    """
    if isinstance(value, dict) and isinstance(value.get("messages"), list):
        messages = value["messages"]
        if messages and isinstance(messages[0], list):  # generation inputs nest one level
            messages = messages[0]
        return messages
    if isinstance(value, list) and value and all(isinstance(m, dict) for m in value):
        return value
    return None


# ---------------------------------------------------------------------------
# envelope unwrapping
# ---------------------------------------------------------------------------


def _unwrap_command(value: Any) -> list[Any] | None:
    """``Command{graph, update, resume, goto}`` (or a list of update-chunks) -> its messages."""
    chunks = value if isinstance(value, list) else [value]
    messages: list[Any] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            return None
        if "update" in chunk and isinstance(chunk["update"], dict):
            messages.extend(chunk["update"].get("messages") or [])
        elif isinstance(chunk.get("messages"), list):
            messages.extend(chunk["messages"])
        else:
            return None
    return messages or None


def _unwrap_llm_result(value: Any) -> list[Any] | None:
    """``LLMResult{generations: [[{message: ...}]]}`` -> the generation messages."""
    if not (isinstance(value, dict) and isinstance(value.get("generations"), list)):
        return None
    first = value["generations"][0] if value["generations"] else []
    return [g["message"] for g in first if isinstance(g, dict) and "message" in g] or None


# ---------------------------------------------------------------------------
# per-class transforms (each returns the new JSON string, or raises)
# ---------------------------------------------------------------------------


def _render_messages(value: Any) -> str:
    messages = _messages_of(value)
    if messages is None:
        raise ValueError("no message list found")
    return json.dumps(_to_openai(_decode_messages(messages)))


def _render_last_message(value: Any) -> str:
    messages = _messages_of(value)
    if not messages:
        raise ValueError("no message list found")
    return json.dumps(_to_openai(_decode_messages([messages[-1]])))


def _render_triggering_call(value: Any) -> str:
    """The assistant message carrying ``tool_calls``, searched backward through state."""
    messages = _messages_of(value)
    if not messages:
        raise ValueError("no message list found")
    for message in reversed(_decode_messages(messages)):
        if getattr(message, "tool_calls", None):
            return json.dumps(_to_openai([message]))
    raise ValueError("no tool-calling assistant message in state")


def _render_command(value: Any) -> str:
    messages = _unwrap_command(value)
    if messages is None:
        raise ValueError("not a Command/update-chunk envelope")
    return json.dumps(_to_openai(_decode_messages(messages)))


def _render_llm_result(value: Any) -> str:
    messages = _unwrap_llm_result(value)
    if messages is None:
        raise ValueError("not an LLMResult")
    return json.dumps(_to_openai(_decode_messages(messages)))


def _render_tool_output(value: Any) -> str:
    """A tool span's output: the ToolMessage's content, not its serialization envelope."""
    if isinstance(value, dict) and value.get("type") == "tool" and isinstance(value.get("data"), dict):
        content = value["data"].get("content")
        return content if isinstance(content, str) else json.dumps(content)
    messages = _unwrap_command(value)
    if messages:
        decoded = _decode_messages(messages)
        content = getattr(decoded[-1], "content", None)
        return content if isinstance(content, str) else json.dumps(content)
    raise ValueError("not a tool envelope")


# ---------------------------------------------------------------------------
# span classification + the adapter
# ---------------------------------------------------------------------------


def _langgraph_node_of(attrs: dict[str, Any]) -> str | None:
    raw = attrs.get(METADATA)
    if not isinstance(raw, str):
        return None
    try:
        metadata = json.loads(raw)
    except ValueError:
        return None
    node = metadata.get("langgraph_node") if isinstance(metadata, dict) else None
    return node if isinstance(node, str) else None


def _parse(raw: Any) -> Any:
    if not isinstance(raw, str):
        raise ValueError("value is not a string")
    return json.loads(raw)


def render_span(span: ReadableSpan) -> ReadableSpan:
    """Apply the render contract to one span. Never mutates; never raises past export."""
    attrs = dict(span.attributes or {})
    kind = attrs.get(SPAN_KIND)
    if kind not in (*_CHAIN_KINDS, "LLM", "TOOL"):
        return span

    node = _langgraph_node_of(attrs)
    is_node = kind in _CHAIN_KINDS and node is not None and span.name == node
    is_root = kind in _CHAIN_KINDS and not is_node
    failed = False

    def convert(key: str, mime_key: str, fn, value: Any) -> None:
        nonlocal failed
        try:
            attrs[key] = fn(value)
            attrs[mime_key] = JSON_MIME
        except Exception:
            failed = True
            logger.warning("render adapter: %s conversion failed on span %r", key, span.name, exc_info=True)

    if is_node:
        # FAIL-CLOSED dedup: the full-state input copy is removed before conversion is
        # attempted, so no conversion error can put it back on the wire.
        raw_input = attrs.pop(INPUT_VALUE, None)
        attrs.pop(INPUT_MIME, None)
        if span.name == "tools" and raw_input is not None:
            try:
                attrs[INPUT_VALUE] = _render_triggering_call(_parse(raw_input))
                attrs[INPUT_MIME] = JSON_MIME
            except Exception:
                failed = True  # input stays dropped
                logger.warning("render adapter: triggering-call derivation failed on %r", span.name, exc_info=True)
        if OUTPUT_VALUE in attrs:
            try:
                value = _parse(attrs[OUTPUT_VALUE])
                fn = _render_command if _unwrap_command(value) else _render_messages
                convert(OUTPUT_VALUE, OUTPUT_MIME, fn, value)
            except Exception:
                failed = True

    elif is_root:
        if INPUT_VALUE in attrs:
            try:
                convert(INPUT_VALUE, INPUT_MIME, _render_messages, _parse(attrs[INPUT_VALUE]))
            except Exception:
                failed = True
        if OUTPUT_VALUE in attrs:
            try:
                convert(OUTPUT_VALUE, OUTPUT_MIME, _render_last_message, _parse(attrs[OUTPUT_VALUE]))
            except Exception:
                failed = True

    elif kind == "LLM":
        if INPUT_VALUE in attrs:
            try:
                convert(INPUT_VALUE, INPUT_MIME, _render_messages, _parse(attrs[INPUT_VALUE]))
            except Exception:
                failed = True
        if OUTPUT_VALUE in attrs:
            try:
                convert(OUTPUT_VALUE, OUTPUT_MIME, _render_llm_result, _parse(attrs[OUTPUT_VALUE]))
            except Exception:
                failed = True

    elif kind == "TOOL":
        if OUTPUT_VALUE in attrs:
            try:
                rendered = _render_tool_output(_parse(attrs[OUTPUT_VALUE]))
                attrs[OUTPUT_VALUE] = rendered
                attrs.pop(OUTPUT_MIME, None)
            except Exception:
                failed = True
                logger.warning("render adapter: tool output unwrap failed on %r", span.name, exc_info=True)

    if failed:
        attrs[RENDER_FAILED] = True
    return _RenderedSpan(span, attrs)


class _RenderedSpan:
    """Delegates everything to the finished span except the transformed attributes."""

    __slots__ = ("_inner", "_attrs")

    def __init__(self, inner: ReadableSpan, attrs: dict[str, Any]) -> None:
        self._inner = inner
        self._attrs = attrs

    @property
    def attributes(self) -> dict[str, Any]:
        return self._attrs

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class LangfuseRenderExporter(SpanExporter):
    """Wraps the Langfuse OTLP exporter; renders each span, exports the original on error."""

    def __init__(self, wrapped: SpanExporter) -> None:
        self._wrapped = wrapped

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        rendered: list[ReadableSpan] = []
        for span in spans:
            try:
                rendered.append(render_span(span))
            except Exception:
                logger.exception("render adapter: unexpected failure; exporting span %r untouched", span.name)
                rendered.append(span)
        return self._wrapped.export(rendered)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._wrapped.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._wrapped.shutdown()
