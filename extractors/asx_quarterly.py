# extractors/asx_quarterly.py
import os
import json
import logging
from pathlib import Path
from typing import List, Any
from unstructured.partition.pdf import partition_pdf

import spacy
from flashtext import KeywordProcessor

from .base_extractor import BaseExtractor
from models.catalyst_disclosure import (
    CatalystDisclosure,
    Impact,
    Tone,
    ForecastType,
    FilingType,
    Entity
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load spaCy once
NLP = spacy.load("en_core_web_sm")

# ============================================================
# QUARTERLY-SPECIFIC CONFIG
CORE_KEYWORDS = {
    "timing_and_immediacy": [
        "imminent","near-term", "upcoming", "expected shortly", "anticipated",
        "targeted for", "inbound interest", "execution phase"
    ],
    "contractual_catalysts": [
        "agreement", "binding", "term sheets", "contracts", "pending", "negotiation",
        "renewal", "submitted", "proposal","acquisition", "partnership", "discussions",
        "expected to commence","finalizing", "tender"
    ],
    "forward_looking_hints": [
        "anticipate", "expect", "outlook","projected", "forecasted", "opportunities", 
        "pipeline", "strategic review", "advanced discussions", "expansion", "significant"
    ],
    "Regulatory & Compliance": [
        "clearance", "licensing", "approval", "deployment", "advanced",
        "submission", "regulatory approval", "FDA", "TGA", "review", "assay results"
    ]
}

EXCLUDE_HEADINGS = {
    "Tenement Interest Notes:",
    "Competent Person’s Statement"
}

STOP_TRIGGER = "quarterly cash flow report"

# Build FlashText keyword processor
keyword_processor = KeywordProcessor(case_sensitive=False)
for group, kws in CORE_KEYWORDS.items():
    for kw in kws:
        keyword_processor.add_keyword(kw, group)

# ============================================================
class ASXQuarterlyExtractor(BaseExtractor):
    def __init__(self, model_name: str = "gpt-5-nano", llm_client: Any = None, debug: bool = True):
        super().__init__(model_name=model_name, llm_client=llm_client)
        self.debug = debug

    def _parse_sections(self, file_path: str):
        elems = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=False,
            extract_images_in_pdf=False,
        )

        sections, current, heading = [], [], "Unknown"
        for elem in elems:
            if elem.category == "Title":
                if current:
                    sections.append({"heading": heading, "text": "\n".join(current)})
                    current = []
                heading = elem.text.strip()
            elif elem.category in {"NarrativeText", "ListItem"}:
                t = elem.text.strip()
                if t:
                    current.append(t)
        if current:
            sections.append({"heading": heading, "text": "\n".join(current)})

        # Stop at cash flow report
        filtered = []
        for s in sections:
            if STOP_TRIGGER.lower() in s["heading"].lower():
                break
            filtered.append(s)

        # Drop excluded headings
        final_sections = [s for s in filtered if s["heading"].strip() not in EXCLUDE_HEADINGS]
        return final_sections

    def _extract_candidates_spacy_flashtext(self, sections: List[dict]) -> List[str]:
        all_text = "\n".join(s["text"] for s in sections)
        doc = NLP(all_text)
        candidates = []
        for sent in doc.sents:
            s = sent.text.strip()
            if s and keyword_processor.extract_keywords(s):
                candidates.append(s)
        return list(dict.fromkeys(candidates))  # dedupe, preserve order

    def _prompt_pass2(self, numbered_items: str) -> str:
        allowed = ", ".join(f'"{ft.value}"' for ft in ForecastType)
        return f"""
You are an expert financial analyst. 
You will receive a numbered list of candidate sentences extracted from a company's report.

Task:
- From the input sentences, KEEP ONLY those that are true forward-looking statements (plans, projections, forecasts, upcoming actions, regulatory submissions, pending deals, milestones, deployments, approvals, or explicitly scheduled future events).
- DROP sentences that only describe present or past facts, are vague, or offer no actionable forward-looking insight.
- For each KEPT sentence, output a JSON object with the following fields:
  - text: the original sentence (string)
  - impact: one of ["LOW","MED","HIGH"]
  - tone: one of ["positive","neutral","cautious"]
  - forecast_type: one of [{allowed}]
  - score: integer between 1 and 10
  - entities: a list of short strings

Requirements:
- Output MUST be a single JSON array of objects.
- No explanations, no markdown.
- Keep the sentence text EXACTLY as in input.

Input sentences:
{numbered_items}

Return ONLY the JSON array.
""".strip()

    def extract(self, file_path: str, metadata: dict) -> List[CatalystDisclosure]:
        sections = self._parse_sections(file_path)
        if not sections:
            return []

        candidates = self._extract_candidates_spacy_flashtext(sections)
        if self.debug:
            print(f"\n=== PASS-1 CANDIDATES ({len(candidates)}) ===")
            for c in candidates:
                print(" -", c)
            print("=============================================\n")

        if not candidates:
            return []

        # ===================================================================
        # YOUR EXACT BATCHING LOGIC — FULLY IMPLEMENTED
        # ===================================================================
        n = len(candidates)

        if n <= 10:
            num_batches = 1
        elif n < 30:
            num_batches = 2
        elif n < 50:
            num_batches = 3
        elif n < 70:
            num_batches = 5
        elif n < 80:
            num_batches = 6
        elif n < 90:
            num_batches = 7
        elif n < 100:
            num_batches = 8
        else:
            num_batches = 9

        # Create evenly sized batches
        batch_size = max(1, (n + num_batches - 1) // num_batches)  # ceil division
        batches = [
            candidates[i:i + batch_size]
            for i in range(0, n, batch_size)
        ]
        batches = batches[:num_batches]  # enforce max batches

        if self.debug:
            sizes = [len(b) for b in batches]
            print(f"→ {n} candidates → {len(batches)} batches: {sizes}")

        # ===================================================================

        results = []
        global_idx = 1

        for batch_num, batch in enumerate(batches, start=1):
            if not batch:
                continue

            if self.debug:
                print(f"\n--- Batch {batch_num}/{len(batches)} ({len(batch)} sentences) ---")

            numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(batch))
            prompt = self._prompt_pass2(numbered)
            resp = self._ask_llm(prompt)
            json_block = self._extract_json_block(resp)
            items = self._safe_json_load(json_block) or []

            for item in items:
                try:
                    raw_type = item.get("forecast_type", "").lower()
                    if "contract" in raw_type:
                        forecast = ForecastType.CONTRACTUAL
                    elif "regul" in raw_type:
                        forecast = ForecastType.REGULATORY
                    elif "time" in raw_type or "sched" in raw_type:
                        forecast = ForecastType.TIMING
                    else:
                        forecast = ForecastType.HINTS

                    model = CatalystDisclosure(
                        doc_id=metadata.get("doc_id") or Path(file_path).stem,
                        exchange="ASX",
                        filing_type=FilingType.ASX_QUARTERLY,
                        filing_date=metadata.get("date"),
                        source_file=file_path,
                        sentence_id=f"s{global_idx}",
                        text=item.get("text", ""),
                        forward_looking=True,
                        forecast_type=forecast,
                        tone=Tone(item.get("tone", "neutral")),
                        impact=Impact(item.get("impact", "MED")),
                        score=int(item.get("score", 5)),
                        categories_matched=[],
                        entities=[Entity(type="entity", value=e, text=e) for e in item.get("entities", [])],
                    )
                    results.append(model)
                    global_idx += 1

                except Exception as e:
                    if self.debug:
                        print(f"Error parsing item in batch {batch_num}: {e}")
                    continue

        print(f"Total forward-looking statements extracted: {len(results)}")
        if self.debug:
            print(f"\n=== Extraction completed for {metadata.get('doc_id', Path(file_path).stem)} ===\n")

        return results