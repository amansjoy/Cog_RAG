"""Query-time retrieval and generation.

Two answer functions:
  cog_rag_answer     - the full two-stage pipeline (theme → entity diffusion).
  vanilla_rag_answer - top-k chunks by cosine, for A/B comparison.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from openai import AzureOpenAI

from cog_rag.index import CogRagIndex
from cog_rag.llm import chat, cosine_matrix, embed


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """You are a careful clinical-mechanistic explainer. Answer the user's
question using ONLY the evidence below. Trace the cause-effect chains explicitly.
If a relevant link is missing in the evidence, say so rather than invent it.

ACTIVATED THEMES (top-down context):
{themes}

ACTIVATED HIGH-ORDER RELATIONS (multi-entity cause-effect chains):
{relations}

SUPPORTING PASSAGES:
{passages}

QUESTION:
{question}

Write a structured answer:
1. A 2-3 sentence executive summary.
2. The main cause-effect chains, named and described.
3. A short note on what is consistent across the evidence and any gaps.
"""

VANILLA_PROMPT = """Answer using ONLY the evidence below.

EVIDENCE:
{passages}

QUESTION: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Cog-RAG
# ---------------------------------------------------------------------------

def cog_rag_answer(
    client: AzureOpenAI,
    index: CogRagIndex,
    question: str,
    top_k_themes: int = 4,
    seed_k: int = 6,
    hops: int = 2,
    max_passages: int = 8,
) -> dict[str, Any]:
    """Run Stage 1 + Stage 2 + generation. Returns a dict with all intermediate
    artifacts so you can inspect exactly what the system retrieved."""
    q_emb = embed(client, [question])[0]

    # Stage 1: theme activation
    themes, candidate_chunks = index.theme_hg.activate(q_emb, top_k=top_k_themes)

    # Stage 2: entity diffusion within theme-aligned region
    edge_ids = index.entity_hg.diffuse(q_emb, candidate_chunks, seed_k=seed_k, hops=hops)
    relations = [index.entity_hg.edges[i] for i in edge_ids]

    # Pick a manageable set of passages for the generator
    passage_ids = sorted(
        {r.chunk_id for r in relations} | set(candidate_chunks)
    )[:max_passages]
    passages = [index.chunks[cid] for cid in passage_ids]

    rels_block = "\n".join(
        f"- ({', '.join(r.entities)}) :: {r.relation}" for r in relations
    ) or "(none)"
    pass_block = "\n\n---\n\n".join(passages) if passages else "(none)"

    answer = chat(
        client,
        [
            {"role": "system", "content": "You explain mechanisms precisely and avoid speculation."},
            {"role": "user", "content": ANSWER_PROMPT.format(
                themes=", ".join(themes) or "(none)",
                relations=rels_block,
                passages=pass_block,
                question=question,
            )},
        ],
    )
    return {
        "answer": answer,
        "themes": themes,
        "relations": [(r.entities, r.relation, r.chunk_id) for r in relations],
        "chunk_ids": passage_ids,
    }


# ---------------------------------------------------------------------------
# Vanilla baseline
# ---------------------------------------------------------------------------

def vanilla_rag_answer(
    client: AzureOpenAI,
    index: CogRagIndex,
    question: str,
    k: int = 5,
) -> dict[str, Any]:
    q_emb = embed(client, [question])[0]
    sims = cosine_matrix(q_emb, index.chunk_embeddings)
    top = np.argsort(-sims)[:k].tolist()
    passages = [index.chunks[i] for i in top]
    answer = chat(
        client,
        [{"role": "user", "content": VANILLA_PROMPT.format(
            passages="\n\n---\n\n".join(passages),
            question=question,
        )}],
    )
    return {"answer": answer, "chunk_ids": top}
