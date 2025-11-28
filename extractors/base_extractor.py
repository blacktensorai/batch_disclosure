# extractors/base_extractor.py
import abc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from utils.tracking import track_tokens
from utils.llm import get_llm, _retry_call
from langchain_core.messages import HumanMessage

# Exceptions for extractors
logger = logging.getLogger(__name__)

class ExtractionError(Exception):
    """Raised when LLM or extraction logic fails."""

class InvalidFilingError(Exception):
    """Raised when a document is invalid or unreadable."""


# ============================================================
# BASE EXTRACTOR
class BaseExtractor(abc.ABC):
    """
    Base class for all extractors:
      - ASX Annual
      - ASX Quarterly
      - ASX Investor Presentation

    Provides:
      - unified file reading
      - LLM querying
      - JSON extraction utilities
      - safe run() wrapper
    """

    def __init__(self, model_name: str = "gpt-5-nano", llm_client=None):
        self.model_name = model_name
        self.llm = llm_client or get_llm(model_name)

    @abc.abstractmethod
    def extract(self, file_path: str, metadata: Dict[str, Any]) -> List[BaseModel]:
        raise NotImplementedError

    # ------------------------------------------------------------
    # File Reading
    def _read_file(self, path: str, encoding: str = "utf-8") -> Any:
        """Read text/HTML as string or return path for PDFs."""

        p = Path(path)
        if not p.exists():
            raise InvalidFilingError(f"File not found: {path}")

        suffix = p.suffix.lower()

        try:
            # PDFs → return file path (to be handled by partition_pdf or PDF parser)
            if suffix == ".pdf":
                logger.debug(f"PDF detected — returning path only: {path}")
                return str(p)

            # Text or HTML
            elif suffix in [".html", ".htm", ".txt"]:
                return p.read_text(encoding=encoding)

            # Fallback to bytes
            else:
                return p.read_bytes()

        except Exception as e:
            raise InvalidFilingError(f"Failed to read file {path}") from e

    # ------------------------------------------------------------
    # LLM Invocation Wrapper
    def _ask_llm(self, prompt: str) -> str:
        """Unified LLM invocation with token tracking and safe fallback."""
        try:
            resp = _retry_call(lambda: self.llm.invoke([HumanMessage(content=prompt)]), retries=3, delay=1.5)
            # token tracking
            try:
                track_tokens(resp)
            except Exception:
                pass

            # normal case
            if hasattr(resp, "content") and resp.content:
                return resp.content

            return str(resp)

        except Exception as e:
            logger.error(f"LLM error → {e}")
            raise ExtractionError("LLM call failed") from e

    # ------------------------------------------------------------
    # JSON Helpers
    def _extract_json_block(self, text: str) -> Optional[str]:
        """Extracts a JSON array [ ... ] from messy LLM output."""
        import re

        if not text:
            return None

        cleaned = text.strip()
        cleaned = re.sub(r"^```json|```$", "", cleaned, flags=re.IGNORECASE)

        s, e = cleaned.find("["), cleaned.rfind("]")
        if s == -1 or e == -1:
            return None

        block = cleaned[s: e + 1]
        block = re.sub(r",\s*}", "}", block)
        block = re.sub(r",\s*]", "]", block)
        return block

    def _safe_json_load(self, text: Optional[str]):
        """Safely parse JSON, returning None on failure."""
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return None

    # ------------------------------------------------------------
    # Main Execution Wrapper
    def run(self, file_path: str, metadata: Dict[str, Any]) -> List[BaseModel]:
        """
        Entry point used by the pipeline.
        Wraps subclass extraction with unified error handling.
        """
        try:
            return self.extract(file_path, metadata)

        except InvalidFilingError as e:
            logger.warning(f"Invalid filing: {metadata.get('doc_id')} — {e}")
            return []

        except ExtractionError as e:
            logger.error(f"Extractor failure: {metadata.get('doc_id')} — {e}")
            return []

        except Exception as e:
            logger.exception(f"Unexpected extractor error: {e}")
            return []