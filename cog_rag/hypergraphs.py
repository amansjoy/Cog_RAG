"""The two hypergraphs: theme (global) and entity (local high-order).

This is the core novelty of Cog-RAG. Read this file alongside the README
section "The two hypergraphs" — every concept there has a counterpart here.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from openai import AzureOpenAI

from cog_rag.llm import cosine_matrix, embed


# ---------------------------------------------------------------------------
# Theme hypergraph: each hyperedge is a theme connecting all chunks it tags
# ---------------------------------------------------------------------------

@dataclass
class ThemeHypergraph:
    theme_to_chunks: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    themes: list[str] = field(default_factory=list)
    embeddings: np.ndarray | None = None  # (n_themes, d)

    def add(self, chunk_id: int, themes: Iterable[str]) -> None:
        for t in themes:
            t_norm = t.strip().lower()
            if t_norm:
                self.theme_to_chunks[t_norm].add(chunk_id)

    def finalize(self, client: AzureOpenAI) -> None:
        """Lock the theme set and embed each theme string for retrieval."""
        self.themes = sorted(self.theme_to_chunks.keys())
        self.embeddings = embed(client, self.themes)

    def activate(self, query_emb: np.ndarray, top_k: int = 4) -> tuple[list[str], set[int]]:
        """Stage 1 of Cog-RAG.

        Pick the top_k themes most similar to the query and return:
          - the themes themselves (for the answer prompt)
          - the union of their member chunks (the candidate set Stage 2 works inside)
        """
        if self.embeddings is None or self.embeddings.shape[0] == 0:
            return [], set()
        sims = cosine_matrix(query_emb, self.embeddings)
        top_idx = np.argsort(-sims)[:top_k]
        top_themes = [self.themes[i] for i in top_idx]
        candidate_chunks: set[int] = set()
        for t in top_themes:
            candidate_chunks.update(self.theme_to_chunks[t])
        return top_themes, candidate_chunks


# ---------------------------------------------------------------------------
# Entity hypergraph: each hyperedge is a multi-entity cause-effect relation
# ---------------------------------------------------------------------------

@dataclass
class EntityHyperedge:
    entities: list[str]
    relation: str
    chunk_id: int


@dataclass
class EntityHypergraph:
    """The "high-order" part. ONE hyperedge can hold 3+ entities at once,
    keeping a multi-entity cause-effect chain intact in a single object.
    A regular graph would shatter that into pairwise edges and lose the
    joint structure."""
    edges: list[EntityHyperedge] = field(default_factory=list)
    entity_to_edges: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    embeddings: np.ndarray | None = None  # (n_edges, d)

    def add(self, entities: list[str], relation: str, chunk_id: int) -> None:
        clean = [e.strip().lower() for e in entities if e and e.strip()]
        if len(clean) < 2 or not relation:
            return
        edge_id = len(self.edges)
        self.edges.append(EntityHyperedge(entities=clean, relation=relation, chunk_id=chunk_id))
        for e in clean:
            self.entity_to_edges[e].add(edge_id)

    def finalize(self, client: AzureOpenAI) -> None:
        texts = [", ".join(e.entities) + " :: " + e.relation for e in self.edges]
        self.embeddings = embed(client, texts) if texts else np.zeros((0, 1), dtype=np.float32)

    def diffuse(
        self,
        query_emb: np.ndarray,
        candidate_chunk_ids: set[int],
        seed_k: int = 6,
        hops: int = 2,
        max_per_hop: int = 8,
    ) -> list[int]:
        """Stage 2 of Cog-RAG.

        Inside the chunks selected by Stage 1, find seed hyperedges most
        similar to the query, then diffuse through the entity hypergraph by
        following shared entities for `hops` rounds. Returns activated edge
        indices in activation order.
        """
        if self.embeddings is None or self.embeddings.shape[0] == 0:
            return []

        sims = cosine_matrix(query_emb, self.embeddings)
        # Restrict to hyperedges that came from theme-aligned chunks.
        mask = np.array(
            [edge.chunk_id in candidate_chunk_ids for edge in self.edges],
            dtype=bool,
        )
        sims_masked = np.where(mask, sims, -np.inf)

        seed_idx = np.argsort(-sims_masked)[:seed_k].tolist()
        seed_idx = [i for i in seed_idx if sims_masked[i] != -np.inf]

        activated: list[int] = list(seed_idx)
        activated_set: set[int] = set(seed_idx)

        frontier_entities: set[str] = set()
        for i in seed_idx:
            frontier_entities.update(self.edges[i].entities)

        for _ in range(hops):
            candidates: list[tuple[int, float]] = []
            for ent in frontier_entities:
                for eid in self.entity_to_edges.get(ent, ()):
                    if eid in activated_set or not mask[eid]:
                        continue
                    candidates.append((eid, float(sims[eid])))
            candidates.sort(key=lambda x: -x[1])
            new_this_hop = [eid for eid, _ in candidates[:max_per_hop]]
            if not new_this_hop:
                break
            new_entities: set[str] = set()
            for eid in new_this_hop:
                activated.append(eid)
                activated_set.add(eid)
                new_entities.update(self.edges[eid].entities)
            frontier_entities = new_entities

        return activated
