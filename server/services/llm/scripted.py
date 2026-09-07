"""A scripted :class:`LLMClient` for tests and the evaluation harness.

Model calls are the only source of non-determinism in OpenPoke's orchestration
layer. Replace them with a fixed script and everything else -- the tool loop,
the iteration cap, batching, error handling, the roster shortlist -- becomes
ordinary deterministic code that can be tested in milliseconds.

Two ways to drive it:

*Queued responses*, when you know the exact sequence you want::

    llm = ScriptedLLM([
        tool_call_response(("send_message_to_user", {"message": "on it"})),
        text_response("done"),
    ])

*A handler*, when the response should depend on the request (the simulation
harness uses this to answer classifier calls from ground-truth labels)::

    llm = ScriptedLLM(handler=lambda req: text_response("important"
                      if "URGENT" in req.messages[-1]["content"] else "skip"))

Queued responses are consumed first; the handler covers anything left over.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------
# Response builders
# --------------------------------------------------------------------------

def text_response(content: str) -> Dict[str, Any]:
    """A plain assistant reply with no tool calls -- ends an agent loop."""

    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def tool_call_response(
    *calls: Tuple[str, Dict[str, Any]],
    content: str = "",
) -> Dict[str, Any]:
    """An assistant reply requesting one or more tool calls.

    Pass several to model parallel tool calls, which is how the interaction
    agent fans work out across execution agents::

        tool_call_response(
            ("send_message_to_agent", {"agent_name": "Email to Alice", ...}),
            ("send_message_to_agent", {"agent_name": "Email to Bob", ...}),
        )
    """

    tool_calls = [
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
        for index, (name, arguments) in enumerate(calls, start=1)
    ]
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}
        ]
    }


def raw_tool_call_response(*tool_calls: Dict[str, Any], content: str = "") -> Dict[str, Any]:
    """Escape hatch for malformed payloads -- unparseable arguments, missing
    names, wrong types. Used to test that the runtimes degrade gracefully
    instead of crashing."""

    return {
        "choices": [
            {"message": {"role": "assistant", "content": content, "tool_calls": list(tool_calls)}}
        ]
    }


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RecordedCall:
    """One request the system under test made to the model."""

    model: str
    messages: List[Dict[str, Any]]
    system: Optional[str]
    tools: Optional[List[Dict[str, Any]]]

    @property
    def tool_names_offered(self) -> List[str]:
        return [
            (tool.get("function") or {}).get("name", "")
            for tool in (self.tools or [])
        ]

    @property
    def prompt_chars(self) -> int:
        """Cheap proxy for prompt size.

        Character count rather than tokens on purpose: it needs no tokenizer, it
        is exactly reproducible, and for the question we actually ask of it --
        *does the prompt grow as the roster grows?* -- the constant factor
        between characters and tokens does not matter.
        """

        total = len(self.system or "")
        for message in self.messages:
            content = message.get("content")
            if isinstance(content, str):
                total += len(content)
            elif content is not None:
                total += len(json.dumps(content, default=str))
        return total


class ScriptExhausted(AssertionError):
    """Raised when the system under test asked for more responses than scripted.

    An assertion rather than a plain error: an exhausted script almost always
    means the code took an unexpected path, and that should read as a test
    failure, not an infrastructure problem.
    """


# --------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------

class ScriptedLLM:
    """An :class:`LLMClient` that replays a fixed script and records requests."""

    def __init__(
        self,
        responses: Optional[Sequence[Dict[str, Any]]] = None,
        *,
        by_model: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None,
        handler: Optional[Callable[[RecordedCall], Dict[str, Any]]] = None,
    ) -> None:
        self._queue: List[Dict[str, Any]] = list(responses or [])
        self._by_model: Dict[str, List[Dict[str, Any]]] = {
            key: list(value) for key, value in (by_model or {}).items()
        }
        self._handler = handler
        self.calls: List[RecordedCall] = []

    # -- scripting ---------------------------------------------------------

    def queue(self, *responses: Dict[str, Any]) -> "ScriptedLLM":
        """Append to the default queue. Chainable."""

        self._queue.extend(responses)
        return self

    def queue_for(self, model_fragment: str, *responses: Dict[str, Any]) -> "ScriptedLLM":
        """Append to a queue matched by substring against the request's model.

        Useful when two agents run in one test and you want each to have its own
        script -- point them at different models in settings, then script each
        by name.
        """

        self._by_model.setdefault(model_fragment, []).extend(responses)
        return self

    # -- the LLMClient protocol -------------------------------------------

    async def complete(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        api_key: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        call = RecordedCall(
            model=model,
            messages=[dict(message) for message in messages],
            system=system,
            tools=list(tools) if tools else None,
        )
        self.calls.append(call)

        for fragment, queued in self._by_model.items():
            if fragment in model and queued:
                return queued.pop(0)

        if self._queue:
            return self._queue.pop(0)

        if self._handler is not None:
            return self._handler(call)

        raise ScriptExhausted(
            f"ScriptedLLM ran out of responses on call #{len(self.calls)} "
            f"(model={model!r}). Either the code under test made more model "
            f"calls than expected, or the script is short by "
            f"{1} response. Calls so far: {self.summary()}"
        )

    # -- assertions --------------------------------------------------------

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def last_request(self) -> RecordedCall:
        if not self.calls:
            raise AssertionError("No model calls were made")
        return self.calls[-1]

    def tool_names_offered(self) -> List[str]:
        """Every distinct tool name offered across all calls, in first-seen order."""

        seen: List[str] = []
        for name in itertools.chain.from_iterable(c.tool_names_offered for c in self.calls):
            if name and name not in seen:
                seen.append(name)
        return seen

    def prompt_chars(self) -> List[int]:
        """Prompt size per call -- the series behind the context-growth metric."""

        return [call.prompt_chars for call in self.calls]

    def find_in_prompt(self, needle: str) -> List[int]:
        """Indices of the calls whose system prompt or messages contain ``needle``."""

        hits: List[int] = []
        for index, call in enumerate(self.calls):
            haystack = (call.system or "") + json.dumps(call.messages, default=str)
            if needle in haystack:
                hits.append(index)
        return hits

    def summary(self) -> str:
        return ", ".join(
            f"#{index + 1} {call.model}" for index, call in enumerate(self.calls)
        ) or "(none)"


__all__ = [
    "ScriptedLLM",
    "ScriptExhausted",
    "RecordedCall",
    "text_response",
    "tool_call_response",
    "raw_tool_call_response",
]
