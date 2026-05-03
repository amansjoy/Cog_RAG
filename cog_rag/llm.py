"""Thin wrappers around Azure OpenAI calls plus a numpy cosine helper.

All functions take an explicit AzureOpenAI client so they are trivially
testable (you can inject a mock or a recording client).
"""
from __future__ import annotations

import numpy as np
from openai import AzureOpenAI

from cog_rag.config import get_settings

_EMBED_BATCH = 64


def embed(client: AzureOpenAI, texts: list[str]) -> np.ndarray:
    """Batched embedding call. Returns array of shape (n, d)."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    deployment = get_settings().embed_deployment
    out: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH):
        batch = texts[i:i + _EMBED_BATCH]
        resp = client.embeddings.create(model=deployment, input=batch)
        out.extend(d.embedding for d in resp.data)
    return np.array(out, dtype=np.float32)


def chat(client: AzureOpenAI, messages: list[dict], temperature: float = 0.0) -> str:
    """Single-turn chat completion returning the raw assistant string."""
    deployment = get_settings().chat_deployment
    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def cosine_matrix(query: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Cosine similarity between a single query (d,) and a matrix (n, d)."""
    if mat.size == 0:
        return np.zeros((0,), dtype=np.float32)
    q = query / (np.linalg.norm(query) + 1e-9)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    return m @ q
