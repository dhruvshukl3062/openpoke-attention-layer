"""The production :class:`LLMClient`: a thin adapter over the OpenRouter client.

This deliberately contains no logic of its own. Keeping the adapter empty means
the scripted client used in tests exercises exactly the same code paths in the
runtimes as production does -- the only thing that differs is where the response
payload comes from.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...openrouter_client import request_chat_completion


class OpenRouterClient:
    """Calls the real OpenRouter API."""

    async def complete(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        api_key: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return await request_chat_completion(
            model=model,
            messages=messages,
            system=system,
            api_key=api_key,
            tools=tools,
        )


__all__ = ["OpenRouterClient"]
