"""The CogRagIndex bundles everything needed at query time.

Build it once (slow: many LLM calls during extraction), pickle it, then
load instantly for every subsequent query.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
from openai import AzureOpenAI

from cog_rag.chunking import chunk_text
from cog_rag.extraction import extract
from cog_rag.hypergraphs import EntityHypergraph, ThemeHypergraph
from cog_rag.llm import embed


@dataclass
class CogRagIndex:
    chunks: list[str]
    chunk_embeddings: np.ndarray
    theme_hg: ThemeHypergraph
    entity_hg: EntityHypergraph

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "CogRagIndex":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_index(client: AzureOpenAI, full_text: str, verbose: bool = True) -> CogRagIndex:
    chunks = chunk_text(full_text)
    if verbose:
        print(f"[index] chunks: {len(chunks)}")

    theme_hg = ThemeHypergraph()
    entity_hg = EntityHypergraph()

    for cid, chunk in enumerate(chunks):
        data = extract(client, chunk)
        theme_hg.add(cid, data["themes"])
        for r in data["relations"]:
            entity_hg.add(r.get("entities", []), r.get("relation", ""), cid)
        if verbose and (cid + 1) % 10 == 0:
            print(
                f"[extract] {cid + 1}/{len(chunks)} "
                f"themes={len(theme_hg.theme_to_chunks)} edges={len(entity_hg.edges)}"
            )

    if verbose:
        print(f"[embed] themes={len(theme_hg.theme_to_chunks)} edges={len(entity_hg.edges)}")

    theme_hg.finalize(client)
    entity_hg.finalize(client)
    chunk_embeddings = embed(client, chunks)

    return CogRagIndex(
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        theme_hg=theme_hg,
        entity_hg=entity_hg,
    )
