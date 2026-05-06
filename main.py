from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
import json
# import requests

from pydantic import BaseModel, Field, field_validator
import os
from typing import List, Dict, Optional, Union, Any

app = FastAPI()

GH_PAGES_DOMAIN = os.getenv("GH_PAGES_DOMAIN", "").strip()
allowed_origins = [
    "http://127.0.0.1:5501",
    "http://localhost:5501",
]
if GH_PAGES_DOMAIN:
    allowed_origins.append(f"https://{GH_PAGES_DOMAIN}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

#####################################################################

class Listing(BaseModel):
    listing_type: Optional[str] = None
    condition: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    grade: Optional[str] = None
    platform: Optional[str] = None
    date: Optional[str] = None
    id: Optional[Union[int, str]] = None


class InterpretRequest(BaseModel):
    """
    v1 request contract (no legacy payloads):
    - `market_summary`: compact aggregates + bounded samples (client-generated)
    - `listings_snippet`: optional extra raw rows for grounding (bounded by client)
    """
    schema_version: int = Field(ge=1)

    market_summary: Dict[str, Any]
    listings_snippet: Optional[List[Listing]] = None

    item_context: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    client_capabilities: Optional[Dict[str, Any]] = None


class Interpretation(BaseModel):
    summary: str = Field(description="Interpretation-focused summary including overall range context (not buying advice).")

    plan: str = Field(description="Describe how you'll solve the problem.")
    reasoning_steps: List[str] = Field(description="Reasoning Steps")

    # Per-condition (or grade) plausible trading ranges keyed by strings like VG/VG+/NM/etc.
    grade_chart: Dict[str, List[float]]

    evidence: List[str] = Field(description="List of Evidence Supporting Summary", max_length=8)
    assumptions: List[str] = Field(description="List of Assumptions for Summary", max_length=6)
    limitations: List[str] = Field(description="List of Limitations of Summary", max_length=6)
    alternative_interpretations: List[str] = Field(description="List of Alternative explanations", max_length=8)

    # Optional macro fields compatible with presets / dashboards
    current_estimate: Optional[float] = None
    current_high_range: Optional[float] = None
    current_low_range: Optional[float] = None
    current_trend: str = Field(description="'increasing' or 'decreasing' or 'steady'")

    @field_validator("grade_chart")
    @classmethod
    def validate_grade_chart_pairs(cls, v: Dict[str, List[float]]) -> Dict[str, List[float]]:
        for k, pair in v.items():
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"grade_chart[{k!r}] must be a two-element [low, high] list")
            low, high = float(pair[0]), float(pair[1])
            if low > high:
                low, high = high, low
            v[k] = [low, high]
        return v
        
print("AI Interpreter Loaded")

@app.post("/interpret/")
@app.post("/interpret")
async def interpret(req: InterpretRequest):
    normalized = normalize_request(req)

    google_api_key = os.getenv('GOOGLE_API_KEY')
    if not google_api_key:
        return JSONResponse(status_code=500, content={"error": "GOOGLE_API_KEY is not set"})

    client = genai.Client(api_key=google_api_key)

    prompt = f"""
You are an expert in collectibles with knowledge of market trends.

Task:
Produce an INTERPRETATION of current marketplace signals for ONE item (not purchasing advice).

Inputs (JSON):
{json.dumps(normalized)}

How to use inputs:
- Prefer `market_summary` for your main conclusions.
- Use `listings_snippet` only as clarifying examples (it may be incomplete).
- If data is sparse, say so explicitly in `limitations` and widen ranges cautiously.

Output requirements:
- Output MUST be valid JSON and MUST match the server's JSON schema.
- Include `summary` that states this is interpretation (not a recommendation) and mentions uncertainty when appropriate.
- Include `grade_chart` mapping each relevant condition/grade key -> [low_usd, high_usd].
- `current_trend` must be exactly one of: increasing, decreasing, steady.
- Keep list fields short and readable.
""".strip()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=types.Part.from_text(text=prompt),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=Interpretation,
            ),
        )

        try:
            parsed = Interpretation.model_validate_json(response.text)
        except Exception as e:
            return JSONResponse(
                status_code=502,
                content={"error": f"Model JSON parse/validate failed: {e}", "fallback": True},
            )

        payload = enrich_for_frontend_compat(parsed.model_dump(mode="python"))
        return JSONResponse(content=payload)
    finally:
        client.close()


def normalize_request(req: InterpretRequest) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["schema_version"] = req.schema_version
    out["correlation_id"] = req.correlation_id
    out["client_capabilities"] = req.client_capabilities or {}
    out["item_context"] = req.item_context or {}

    out["market_summary"] = req.market_summary

    if req.listings_snippet:
        out["listings_snippet"] = [l.model_dump(mode="python") for l in req.listings_snippet]

    return out


def enrich_for_frontend_compat(model_out: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(model_out)
    alts = payload.get("alternative_interpretations") or payload.get("alternatives")
    if alts:
        payload["alternatives"] = alts

    gcs = payload.get("grade_chart")
    if gcs and isinstance(gcs, dict):
        normalized_gc: Dict[str, List[float]] = {}
        for k, v in gcs.items():
            try:
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    normalized_gc[str(k)] = [float(v[0]), float(v[1])]
            except Exception:
                continue
        payload["grade_chart"] = normalized_gc

    return payload

