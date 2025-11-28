# pipeline/__init__.py
"""
Pipeline for:
1. Ingesting filings (ASX + SEC)
2. Dispatching correct extractor
3. Producing structured catalysts JSON
"""


from .dispatcher import get_extractor_instance
from .run_pipeline import Pipeline, process_file_request

__all__ = ["get_extractor_instance", "Pipeline", "process_file_request"]
