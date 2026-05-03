"""Theme + high-order entity extraction.

For each chunk we ask the LLM for:
  - 2-4 themes (high-level topics)
  - up to 6 high-order relations, each with 2-5 entities + a sentence

This is the offline knowledge-construction step. It is the slow part of
indexing because it is one LLM call per chunk; it is also where the quality
of the entire system is largely determined.
"""
from __future__ import annotations

import json
import re

from openai import AzureOpenAI

from cog_rag.llm import chat


EXTRACTION_PROMPT = """You will read a passage from a medical review on chronic stress and cardiovascular disease.

Return ONLY a single JSON object with exactly two top-level keys:
  "themes":      list of 2 to 4 short noun phrases (3-6 words each) naming high-level
                 topics the passage discusses. Themes should be reusable across passages
                 (good: "endothelial dysfunction"; bad: "what this paragraph says").
  "relations":   list of HIGH-ORDER cause-effect groups extracted from the passage.
                 Each relation is an object:
                     {"entities": [...], "relation": "..."}
                 - "entities" is a list of 2 to 5 distinct medical entities
                   (molecules, organs, processes, conditions). PREFER 3-5 over 2.
                 - "relation" is a one-sentence summary of how those entities
                   participate in a single cause-effect chain.

Rules:
- Up to 6 relations. Skip the passage entirely (return empty lists) if it is
  filler with no medical content.
- Use canonical names: "cortisol", not "the hormone cortisol".
- Do NOT return prose outside the JSON. Do NOT wrap in markdown.

Passage:
\"\"\"
{passage}
\"\"\"
"""


def extract(client: AzureOpenAI, passage: str) -> dict:
    """Call the LLM and parse the JSON. Resilient to occasional code-fence wrapping."""
    raw = chat(
        client,
        [
            {"role": "system", "content": "You extract structured medical knowledge. Output only valid JSON."},
            {"role": "user", "content": EXTRACTION_PROMPT.format(passage=passage)},
        ],
    ).strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.MULTILINE)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"themes": [], "relations": []}
    data.setdefault("themes", [])
    data.setdefault("relations", [])
    return data
