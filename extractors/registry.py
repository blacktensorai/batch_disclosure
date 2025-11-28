# extractors/registry.py
from extractors.asx_annual import ASXAnnualExtractor
from extractors.asx_quarterly import ASXQuarterlyExtractor
from extractors.asx_investor import ASXInvestorExtractor
from extractors.sec_10q import SECExtractor
from models.enums import FilingType

EXTRACTOR_REGISTRY = {
    ("ASX", FilingType.ASX_ANNUAL): ASXAnnualExtractor,
    ("ASX", FilingType.ASX_QUARTERLY): ASXQuarterlyExtractor,
    ("ASX", FilingType.ASX_INVESTOR): ASXInvestorExtractor,

    ("SEC", FilingType.SEC_10Q): SECExtractor,
}


def get_extractor(exchange: str, filing_type):
    """
    Returns a NEW extractor instance for the given exchange & filing type.
    """
    key = (exchange.upper(), filing_type)

    if key not in EXTRACTOR_REGISTRY:
        raise KeyError(f"No extractor registered for: {key}")

    extractor_cls = EXTRACTOR_REGISTRY[key]
    return extractor_cls()