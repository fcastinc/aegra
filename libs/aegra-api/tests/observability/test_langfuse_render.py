"""Render-adapter contract tests, driven by REAL span shapes recorded from live traces.

Fixtures under fixtures/ were pulled from production Langfuse observations (2026-07-21,
traces 1810931…/0416d02…): the exact wrapped LangChain serializations the adapter exists
to convert. Assertions are the adapter contract (agents/docs/dev-475 spec):
- every converted value parses as OpenAI-format messages
- zero LangChain envelope keys (type/data, graph/update, lc-constructor) survive
- tool_calls carry JSON-string arguments
- node inputs are dropped fail-closed; tools-node input becomes the triggering call
- reasoning re-attaches as Langfuse's registered `thinking` field
- failures pass the raw value through, stamped with aegra.render.failed
"""

import json
from pathlib import Path
from typing import Any

import pytest

from aegra_api.observability.langfuse_render import (
    INPUT_MIME,
    INPUT_VALUE,
    OUTPUT_VALUE,
    RENDER_FAILED,
    SPAN_KIND,
    LangfuseRenderExporter,
    render_span,
)

FIXTURES = Path(__file__).parent / "fixtures"


class Span:
    """Minimal ReadableSpan stand-in: the adapter reads only .name and .attributes."""

    def __init__(self, name: str, attributes: dict[str, Any]):
        self.name = name
        self.attributes = attributes


def fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def span_from_fixture(name: str, kind: str, langgraph_node: str | None = None) -> Span:
    fx = fixture(name)
    attrs: dict[str, Any] = {SPAN_KIND: kind}
    if fx.get("input") is not None:
        attrs[INPUT_VALUE] = json.dumps(fx["input"]) if not isinstance(fx["input"], str) else fx["input"]
    if fx.get("output") is not None:
        attrs[OUTPUT_VALUE] = json.dumps(fx["output"]) if not isinstance(fx["output"], str) else fx["output"]
    metadata = dict(fx.get("metadata") or {})
    if langgraph_node is not None:
        metadata["langgraph_node"] = langgraph_node
    attrs["metadata"] = json.dumps(metadata)
    return Span(fx["name"], attrs)


ENVELOPE_KEYS = {"data", "graph", "update", "resume", "goto", "lc", "kwargs", "generations"}


def assert_openai_messages(raw: str) -> list[dict[str, Any]]:
    """The output contract: a list of OpenAI-format messages, no envelope keys anywhere."""
    messages = json.loads(raw)
    assert isinstance(messages, list) and messages, f"not a message list: {raw[:120]}"
    for message in messages:
        assert isinstance(message, dict) and "role" in message, f"no role: {message}"
        assert not (ENVELOPE_KEYS & set(message)), f"envelope keys leaked: {sorted(ENVELOPE_KEYS & set(message))}"
        for tc in message.get("tool_calls") or []:
            args = tc.get("function", {}).get("arguments", tc.get("arguments"))
            assert isinstance(args, str), f"tool_call arguments not a JSON string: {tc}"
    return messages


# --- graph roots ---------------------------------------------------------------------


def test_root_input_is_openai_messages():
    rendered = render_span(span_from_fixture("root", "CHAIN"))
    messages = assert_openai_messages(rendered.attributes[INPUT_VALUE])
    assert messages[-1]["role"] == "user"


def test_root_output_is_the_answer_only():
    rendered = render_span(span_from_fixture("root", "CHAIN"))
    messages = assert_openai_messages(rendered.attributes[OUTPUT_VALUE])
    assert len(messages) == 1 and messages[0]["role"] == "assistant"


def test_nested_subagent_root_reduces_despite_inherited_langgraph_node():
    # data_load runs inside main's `tools` node and inherits langgraph_node="tools";
    # its name differs, so it must classify as a root, not a node.
    rendered = render_span(span_from_fixture("nested_root", "CHAIN", langgraph_node="tools"))
    assert_openai_messages(rendered.attributes[INPUT_VALUE])
    out = assert_openai_messages(rendered.attributes[OUTPUT_VALUE])
    assert len(out) == 1 and out[0]["role"] == "assistant"


def test_root_reattaches_reasoning_as_thinking():
    fx = {"messages": [{"type": "ai", "data": {
        "content": "the answer", "additional_kwargs": {"reasoning_content": "step by step"}}}]}
    span = Span("unified_assistant", {
        SPAN_KIND: "CHAIN", OUTPUT_VALUE: json.dumps(fx), "metadata": "{}"})
    out = assert_openai_messages(render_span(span).attributes[OUTPUT_VALUE])
    assert out[0]["thinking"] == [{"type": "thinking", "content": "step by step"}]


# --- nodes ---------------------------------------------------------------------------


def full_state_input() -> Any:
    return fixture("nested_root")["input"]  # real {type,data} message-state shape


def test_model_node_input_dropped():
    span = Span("model", {
        SPAN_KIND: "CHAIN",
        INPUT_VALUE: json.dumps(full_state_input()),
        INPUT_MIME: "application/json",
        "metadata": json.dumps({"langgraph_node": "model"}),
    })
    rendered = render_span(span)
    assert INPUT_VALUE not in rendered.attributes
    assert INPUT_MIME not in rendered.attributes


def test_node_input_drop_is_fail_closed_on_garbage():
    span = Span("model", {
        SPAN_KIND: "CHAIN",
        INPUT_VALUE: "{corrupt json",
        "metadata": json.dumps({"langgraph_node": "model"}),
    })
    rendered = render_span(span)
    assert INPUT_VALUE not in rendered.attributes  # dropped even though unparseable


def test_tools_node_input_toolcall_list_form():
    # THE live form (verified via local instrumented run 2026-07-21): LangChain v1's
    # ToolNode input is the tool_calls list itself, not state.
    calls = [{"name": "get_data", "args": {"series": "natgas"}, "id": "call_1", "type": "tool_call"}]
    span = Span("tools", {
        SPAN_KIND: "CHAIN",
        INPUT_VALUE: json.dumps(calls),
        "metadata": json.dumps({"langgraph_node": "tools"}),
    })
    messages = assert_openai_messages(render_span(span).attributes[INPUT_VALUE])
    assert messages[0]["role"] == "assistant"
    tc = messages[0]["tool_calls"][0]
    assert tc["function"]["name"] == "get_data"
    assert json.loads(tc["function"]["arguments"]) == {"series": "natgas"}


def test_empty_middleware_output_not_marked_failed():
    span = Span("TelemetrySummarizationMiddleware.before_model", {
        SPAN_KIND: "CHAIN",
        OUTPUT_VALUE: "{}",
        "metadata": json.dumps({"langgraph_node": "TelemetrySummarizationMiddleware.before_model"}),
    })
    rendered = render_span(span)
    assert rendered.attributes[OUTPUT_VALUE] == "{}"
    assert RENDER_FAILED not in rendered.attributes


def test_tools_node_input_becomes_triggering_call():
    state = {"messages": [
        {"type": "human", "data": {"content": "load it"}},
        {"type": "ai", "data": {
            "content": "",
            "tool_calls": [{"name": "data_access", "args": {"action": "query_data"}, "id": "call_1", "type": "tool_call"}],
        }},
    ]}
    span = Span("tools", {
        SPAN_KIND: "CHAIN",
        INPUT_VALUE: json.dumps(state),
        "metadata": json.dumps({"langgraph_node": "tools"}),
    })
    messages = assert_openai_messages(render_span(span).attributes[INPUT_VALUE])
    assert len(messages) == 1 and messages[0]["role"] == "assistant"
    assert messages[0]["tool_calls"], "triggering call must carry tool_calls"


def test_tools_node_output_chunks_unwrap():
    rendered = render_span(span_from_fixture("tools_node_chunks", "CHAIN", langgraph_node="tools"))
    messages = assert_openai_messages(rendered.attributes[OUTPUT_VALUE])
    assert all(m["role"] == "tool" for m in messages)


def test_tools_node_plain_delta_output_converts():
    rendered = render_span(span_from_fixture("tools_node", "CHAIN", langgraph_node="tools"))
    assert_openai_messages(rendered.attributes[OUTPUT_VALUE])


def test_agent_named_hook_is_still_a_node():
    # OI marks any name containing "agent" as kind=AGENT; the name==langgraph_node
    # test must still classify hook spans as nodes and drop their input.
    span = Span("OrphanedToolCallRepair.before_agent", {
        SPAN_KIND: "AGENT",
        INPUT_VALUE: json.dumps(full_state_input()),
        "metadata": json.dumps({"langgraph_node": "OrphanedToolCallRepair.before_agent"}),
    })
    rendered = render_span(span)
    assert INPUT_VALUE not in rendered.attributes


# --- generations ---------------------------------------------------------------------


def test_generation_output_llm_result_to_assistant_message():
    rendered = render_span(span_from_fixture("generation", "LLM"))
    messages = assert_openai_messages(rendered.attributes[OUTPUT_VALUE])
    assert messages[0]["role"] == "assistant"


def test_generation_input_full_assembly_converts():
    fx = fixture("generation")
    raw = fx.get("input")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not (isinstance(parsed, dict) and parsed.get("messages")):
        # Fork-era artifact: input.value was stripped on LLM spans, so the recorded
        # observation input holds no messages. Stock shape covered by the test below.
        pytest.skip("fixture recorded without generation messages (fork era)")
    rendered = render_span(span_from_fixture("generation", "LLM"))
    assert_openai_messages(rendered.attributes[INPUT_VALUE])


def test_generation_input_stock_shape_converts():
    # Stock OpenInference records run.inputs: {"messages": [[lc-constructor, ...]]}
    def lc(cls, content, extra=None):
        kwargs = {"content": content, "type": {"SystemMessage": "system", "HumanMessage": "human"}[cls]}
        kwargs.update(extra or {})
        return {"lc": 1, "type": "constructor", "id": ["langchain", "schema", "messages", cls], "kwargs": kwargs}

    stock = {"messages": [[lc("SystemMessage", "You are the assistant."), lc("HumanMessage", "load the data")]]}
    span = Span("ReasoningChatOpenAI", {
        SPAN_KIND: "LLM", INPUT_VALUE: json.dumps(stock), "metadata": "{}"})
    messages = assert_openai_messages(render_span(span).attributes[INPUT_VALUE])
    assert [m["role"] for m in messages] == ["system", "user"]


def test_generation_reasoning_reattaches_as_thinking():
    result = {"generations": [[{"text": "hi", "message": {
        "lc": 1, "type": "constructor",
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {"content": "hi", "additional_kwargs": {"reasoning_content": "hmm"}, "type": "ai"},
    }}]]}
    span = Span("ReasoningChatOpenAI", {
        SPAN_KIND: "LLM", OUTPUT_VALUE: json.dumps(result), "metadata": "{}"})
    messages = assert_openai_messages(render_span(span).attributes[OUTPUT_VALUE])
    assert messages[0]["thinking"] == [{"type": "thinking", "content": "hmm"}]


# --- tools ---------------------------------------------------------------------------


def test_task_command_output_unwraps_to_content():
    rendered = render_span(span_from_fixture("task_command", "TOOL"))
    out = rendered.attributes[OUTPUT_VALUE]
    for key in ("graph", "update", '"data"'):
        assert key not in out or not out.startswith("{"), f"envelope survived: {out[:100]}"
    assert "successfully" in out or len(out) > 0  # the subagent's answer text


def test_tool_wrapped_output_unwraps_to_content():
    rendered = render_span(span_from_fixture("tool_wrapped", "TOOL"))
    out = rendered.attributes[OUTPUT_VALUE]
    assert '"data"' not in out.split("content")[0] if "content" in out else True
    assert not out.startswith('{"type": "tool"'), f"ToolMessage envelope survived: {out[:80]}"


def test_tool_input_untouched():
    fx = fixture("tool_wrapped")
    rendered = render_span(span_from_fixture("tool_wrapped", "TOOL"))
    original = json.dumps(fx["input"]) if not isinstance(fx["input"], str) else fx["input"]
    assert rendered.attributes[INPUT_VALUE] == original


# --- safety --------------------------------------------------------------------------


def test_conversion_failure_is_fail_open_and_marked():
    span = Span("unified_assistant", {
        SPAN_KIND: "CHAIN", OUTPUT_VALUE: '{"not": "messages"}', "metadata": "{}"})
    rendered = render_span(span)
    assert rendered.attributes[OUTPUT_VALUE] == '{"not": "messages"}'
    assert rendered.attributes[RENDER_FAILED] is True


def test_unclassified_span_untouched():
    span = Span("whatever", {"some.attr": "x"})
    assert render_span(span) is span


def test_exporter_never_raises(monkeypatch):
    class Boom:
        name = "boom"

        @property
        def attributes(self):
            raise RuntimeError("attribute access explodes")

    class Sink:
        def __init__(self):
            self.received = None

        def export(self, spans):
            self.received = list(spans)
            return "ok"

        def force_flush(self, t=30000):
            return True

        def shutdown(self):
            pass

    sink = Sink()
    exporter = LangfuseRenderExporter(sink)
    boom = Boom()
    assert exporter.export([boom]) == "ok"
    assert sink.received == [boom]  # original span exported untouched
