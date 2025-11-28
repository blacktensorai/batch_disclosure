# utils/tracking.py
"""
Token tracking helper for CatalystScan.
---------------------------------------
- Safe, no-fail token logging
- Works with any LangChain ChatOpenAI-compatible response
- Called by BaseExtractor._ask_llm()
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def _extract_token_usage(response):
    """
    Extract token usage from LangChain/OpenAI response objects.
    Best-effort: returns None if fields do not exist.
    """
    try:
        # OpenAI client style: response.response_metadata["token_usage"]
        meta = getattr(response, "response_metadata", {}) or {}
        usage = meta.get("token_usage") or {}

        if not usage:
            return None

        return {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    except Exception:
        return None


def track_tokens(response):
    """
    Logs token usage for observability.
    Never throws.
    """
    try:
        usage = _extract_token_usage(response)
        if not usage:
            return

        logger.info(
            "LLM Token Usage — time=%s input=%s output=%s total=%s",
            datetime.utcnow().isoformat(),
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("total_tokens"),
        )

    except Exception:
        # Never block main pipeline
        pass
