 # pipeline/dispatcher.py
import logging
from typing import Any
from extractors import registry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_extractor_instance(exchange: str, filing_type: str) -> Any:
    """
    Return the extractor instance for the given exchange and filing_type.
    """
    # Normalize common filing type inputs
    e = exchange.upper()
    ft = filing_type
    # Normalize some synonyms
    mapping = {
        "ANNUAL": "annual",
        "ASX_ANNUAL": "annual",
        "QUARTERLY": "quarterly",
        "ASX_QUARTERLY": "quarterly",
        "INVESTOR_PRESENTATION": "investor",
        "INVESTOR": "investor",
        "10-Q": "10-Q",
        "10Q": "10-Q"
    }

    # canonical filing type
    ft_key = mapping.get(ft.upper(), ft)

    key = (e, ft_key)
    try:
        extractor = registry.get_extractor(e, ft_key)
        logger.info("Dispatcher → found extractor for %s / %s", e, ft_key)
        return extractor
    except KeyError as ke:
        logger.error("XXXXX = No extractor registered for (%s, %s)", e, ft_key)
        raise
