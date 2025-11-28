# models/catalyst_disclosure.py
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from models.enums import Impact, Tone, ForecastType, FilingType


class Entity(BaseModel):
    type: str = "entity"
    value: str
    text: str


class CatalystDisclosure(BaseModel):
    # Document metadata
    doc_id: str
    exchange: str                               # "ASX" or "SEC"
    filing_type: FilingType
    filing_date: Optional[str] = None
    source_file: Optional[str] = None

    # Catalyst content
    sentence_id: str
    text: str                                   # ← NO max_length here
    forward_looking: bool = True

    forecast_type: ForecastType
    tone: Tone
    impact: Impact
    score: int = Field(..., ge=1, le=10)

    # Optional enrichments
    categories_matched: List[str] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    flag: Optional[str] = "ok"

    # ------------------------------------------------------------------
    # AUTOMATIC TRUNCATION FOR DISPLAY / STORAGE (keeps raw text intact)
    # ------------------------------------------------------------------
    @property
    def text_preview(self) -> str:
        """Used only for DataFrames / CSV export – never breaks validation"""
        s = self.text.strip()
        return (s[:380] + "...") if len(s) > 400 else s

    @field_validator("text", mode="before")
    @classmethod
    def clean_text(cls, v: str) -> str:
        if not v:
            return ""
        # Remove excessive newlines/tabs that sometimes come from PDFs
        cleaned = " ".join(line.strip() for line in str(v).splitlines() if line.strip())
        return cleaned

    @field_validator("categories_matched")
    @classmethod
    def dedupe_categories(cls, v):
        return list(dict.fromkeys(v)) if v else []