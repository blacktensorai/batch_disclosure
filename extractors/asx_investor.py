# /extractors/asx_investor.py
import logging
from pathlib import Path
from typing import List, Any
import spacy
from flashtext import KeywordProcessor
from unstructured.partition.pdf import partition_pdf
import re

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
DEBUG = True

# -----------------------------------------------------------
# Load spaCy once
NLP = spacy.load("en_core_web_sm")

# -----------------------------------------------------------
# Investor Keyword Groups
INVESTOR_KEYWORDS = {
    "Intent Verbs": [
        "expected shortly", "expected to", "expect to", "plan to", "planned to",
        "intends to", "intend to", "inbound interest",
        "anticipates", "anticipate", "anticipated",
        "targeting", "targets", "targeted", "finalizing"
    ],
    "Timeline": [
        "over the next", "in FY", "in H1", "in H2", "accelerated", "year end",
        "during", "near-term", "imminent", "upcoming", "expected in",
        "targeted for", "scheduled for", "execution phase"
    ],
    "Guidance": [
        "guidance", "forecast", "outlook", "projected", "opportunities",
        "cash flow positive", "capital raise", "funding secured", "forecasted"
    ],
    "Milestones": [
        "licensing", "clearance", "approval",
        "resource upgrade", "FDA", "deployment", "advanced",
        "regulatory approval", "submission", "tender", "review"
    ],
    "Deals": [
        "agreement expected", "term sheets", "binding", "contracts",
        "MOU", "JV", "partnership", "submitted", "pending", "acquisition",
        "negotiation", "commercial launch", "renewal", "proposal", "discussions",
        "expected to commence"
    ],
    "Strategy": [
        "strategy", "strategic review", "significant", "advanced discussions",
        "expansion", "growth strategy", "finalizing", "pipeline"
    ]
}

CATEGORY_MAP = {
    "Intent Verbs": ForecastType.INTENT,
    "Timeline": ForecastType.TIMING,
    "Guidance": ForecastType.GUIDANCE,
    "Milestones": ForecastType.MILESTONES,
    "Deals": ForecastType.DEALS,
    "Strategy": ForecastType.STRATEGY,
}

# FlashText keyword processor
keyword_processor = KeywordProcessor(case_sensitive=False)
for cat, kws in INVESTOR_KEYWORDS.items():
    for kw in kws:
        keyword_processor.add_keyword(kw, cat)

# -----------------------------------------------------------
# Drop headings
DROP_HEADING_PATTERNS = [
    r"Disclaimer",
    r"Competent\s+Person",
    r"Board|Director|Chairman|CEO|COO|CFO|Management",
    r"Corporate\s+(Snapshot|Overview|Directory|Structure)",
    r"About\s+.*",
    r"Registered\s+Office",
    r"Principal\s+Place",
    r"Investor\s+Relations",
    r"Website",
    r"Financial\s+Snapshot",
    r"Inferred\s+Mineral|JORC|Metallurgy|Assay|Drill|Resource|Geochemistry|Infrastructure",
    r"T\s*cell|CAR[- ]?T|Immune|Mechanism|Safety\s+Profile",
    r"Supply|Demand|Market\s+Fundamentals",
    r"Contact|Appendix|Legal|Notice",
]
DROP_HEADINGS = [re.compile(p, re.IGNORECASE) for p in DROP_HEADING_PATTERNS]

# -----------------------------------------------------------
# Extractor
class ASXInvestorExtractor(BaseExtractor):

    # ------------------------
    def _parse_sections(self, file_path: str) -> List[dict]:
        elems = partition_pdf(
            filename=file_path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=False
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

        # Drop irrelevant headings
        final_sections = [
            s for s in sections
            if not any(p.search(s["heading"]) for p in DROP_HEADINGS)
        ]
        return final_sections

    # ------------------------
    def _extract_candidates(self, sections: List[dict]) -> List[str]:
        all_text = "\n".join(s["text"] for s in sections)
        doc = NLP(all_text)

        candidates = []
        for sent in doc.sents:
            s = sent.text.strip()
            if not s:
                continue
            matches = keyword_processor.extract_keywords(s)
            if matches:
                candidates.append(s)

        return list(dict.fromkeys(candidates))  # preserve order

    # ------------------------
    def _prompt_pass2(self, numbered_items: str) -> str:
        allowed = ", ".join(f'"{ft.value}"' for ft in ForecastType)

        return f"""
You are an expert financial analyst.
You will receive a numbered list of candidate sentences extracted from an ASX investor report.

Task:
- KEEP ONLY true forward-looking statements.
- DROP sentences that describe only past/present facts or vague commentary.
- For each KEPT sentence, output JSON with:
  - text
  - impact (LOW, MED, HIGH)
  - tone (positive, neutral, cautious)
  - forecast_type (one of [{allowed}])
  - score (1–10)
  - entities (list)
  - categories_matched (list)

Output: A single JSON array only.

Input sentences:
{numbered_items}
""".strip()

    # --------------------------------------------------------
    # Extraction pipeline
    def extract(self, file_path: str, metadata: dict) -> List[CatalystDisclosure]:

        sections = self._parse_sections(file_path)
        if not sections:
            return []

        # PASS 1: Candidate extraction
        candidates = self._extract_candidates(sections)

        if DEBUG:
            print(f"\n=== PASS-1 CANDIDATES ({len(candidates)}) ===")
            for c in candidates:
                print(" -", c)
            print("=====================================\n")

        if not candidates:
            return []

        # ----------------------------------------------------
        # Batching logic (same as quarterly)
        n = len(candidates)
        if n <= 10:
            batches = [candidates]
            if DEBUG:
                print(f"Candidates <= 10 → 1 batch with {n} sentences.")
        else:
            first_batch = (n + 1) // 2
            batches = [candidates[:first_batch], candidates[first_batch:]]
            if DEBUG:
                print(f"Candidates > 10 → 2 batches: {len(batches[0])} + {len(batches[1])}")

        # ----------------------------------------------------
        results = []
        global_idx = 1
        doc_id = metadata.get("doc_id") or Path(file_path).stem
        filing_date = metadata.get("date")

        # ----------------------------------------------------
        for batch_num, batch in enumerate(batches, start=1):

            if DEBUG:
                print(f"\n--- Processing batch {batch_num} ({len(batch)} sentences) ---")

            numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(batch))
            prompt = self._prompt_pass2(numbered)

            resp = self._ask_llm(prompt)
            json_block = self._extract_json_block(resp)
            items = self._safe_json_load(json_block) or []

            # Convert each JSON entry to CatalystDisclosure
            for item in items:
                try:
                    cats = item.get("categories_matched", [])
                    if cats:
                        cat_name = cats[0]
                        forecast = CATEGORY_MAP.get(cat_name, ForecastType.STRATEGY)
                    else:
                        forecast = ForecastType.STRATEGY

                    model = CatalystDisclosure(
                        doc_id=doc_id,
                        exchange="ASX",
                        filing_type=FilingType.ASX_INVESTOR,
                        filing_date=filing_date,
                        source_file=file_path,

                        sentence_id=f"s{global_idx}",
                        text=item.get("text", ""),
                        forward_looking=True,

                        forecast_type=forecast,
                        tone=Tone(item.get("tone", "neutral").lower()),
                        impact=Impact(item.get("impact", "MED").upper()),
                        score=int(item.get("score", 5)),

                        categories_matched=cats,
                        entities=[
                            Entity(type="entity", value=str(e), text=str(e))
                            for e in item.get("entities", [])
                        ],
                    )

                    results.append(model)
                    global_idx += 1

                except Exception as e:
                    logger.warning(f"Failed to build model: {e}")
                    continue

        if DEBUG:
            print(f"\nPASS-2 → {len(results)} forward-looking statements extracted.")

        return results
