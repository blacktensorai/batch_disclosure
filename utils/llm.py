# utils/llm.py
import os
import time
import logging
from typing import Optional
from pathlib import Path
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import time

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]  # adjust if deeper nested
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

# ---------------------------------------------
# Retry wrapper (simple but effective)
def _retry_call(fn, retries=3, delay=1.0):
    for i in range(retries):
        try:
            time.sleep(1)
            return fn()
        except Exception as e:
            if i == retries - 1:
                raise
            logger.warning(f"LLM call failed, retrying ({i+1}/{retries}) — {e}")
            time.sleep(delay)

# ---------------------------------------------
# Public Factory
def get_llm(
    model_name: str = "gpt-5-nano",
    temperature: float = 0,
    max_retries: int = 3,
    timeout: Optional[int] = 60,
):
    """
    Returns a ChatOpenAI-compatible LLM client.

    Extractors call:
        self.llm = get_llm(model_name)
    """

    api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")

    # The actual constructor call wrapped in retry
    def build_client():
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )

    try:
        client = _retry_call(build_client, retries=2, delay=0.5)
        logger.info(f"Loaded LLM model: {model_name}")
        return client
    except Exception as e:
        logger.error(f"Failed to load LLM model '{model_name}': {e}")
        raise
