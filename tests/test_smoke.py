"""Smoke tests that don't require Azure credentials.

Run with: pytest tests/   (or right-click in PyCharm)
These exercise the pure-Python parts: chunking, hypergraph data structures,
diffusion logic. The LLM-dependent parts (extraction, retrieval) need
mocking — kept out of this file to stay simple.
"""
from __future__ import annotations

import numpy as np

from cog_rag.chunking import chunk_text
from cog_rag.hypergraphs import EntityHypergraph, ThemeHypergraph


def test_chunk_text_basic():
    text = " ".join(["word"] * 500)
    chunks = chunk_text(text, target_words=100, overlap_words=20)
    assert len(chunks) >= 5
    assert all(len(c.split()) <= 100 for c in chunks)


def test_theme_hypergraph_grouping():
    hg = ThemeHypergraph()
    hg.add(0, ["Cortisol Pathway", "HPA axis"])
    hg.add(1, ["HPA axis"])
    hg.add(2, ["Atherosclerosis"])
    assert hg.theme_to_chunks["hpa axis"] == {0, 1}
    assert hg.theme_to_chunks["cortisol pathway"] == {0}


def test_entity_hypergraph_diffusion_finds_shared_entity():
    """Build a tiny hypergraph by hand and verify that diffusion through a
    shared entity activates a second hyperedge in a single hop."""
    hg = EntityHypergraph()
    # Hyperedge 0 lives in chunk 0 and contains 'cortisol' + 'sodium' + 'bp'
    hg.add(["cortisol", "sodium", "bp"], "cortisol promotes sodium retention raising bp", 0)
    # Hyperedge 1 lives in chunk 1 and contains 'cortisol' + 'inflammation'
    hg.add(["cortisol", "inflammation"], "cortisol affects inflammation", 1)
    # Hyperedge 2 lives in chunk 2 and shares NO entity with the seed
    hg.add(["sleep", "vagal tone"], "sleep affects vagal tone", 2)

    # Inject deterministic embeddings so we control what the seed selection picks.
    # Make hyperedge 0 the closest match to the query, and hyperedge 1 second.
    hg.embeddings = np.array(
        [[1.0, 0.0],   # edge 0: identical to query
         [0.9, 0.1],   # edge 1: close to query
         [0.0, 1.0]],  # edge 2: orthogonal
        dtype=np.float32,
    )
    query = np.array([1.0, 0.0], dtype=np.float32)

    activated = hg.diffuse(
        query_emb=query,
        candidate_chunk_ids={0, 1, 2},
        seed_k=1,    # force only edge 0 as seed
        hops=1,
        max_per_hop=4,
    )
    # Edge 0 is the seed; edge 1 should be reached via shared entity 'cortisol';
    # edge 2 has no shared entity with the seed and must NOT activate.
    assert 0 in activated
    assert 1 in activated
    assert 2 not in activated


def test_diffusion_respects_candidate_set():
    """Hyperedges in chunks outside the theme-aligned candidate set must be
    excluded even if they share an entity with the seeds."""
    hg = EntityHypergraph()
    hg.add(["a", "b"], "edge0 in chunk 0", 0)
    hg.add(["a", "c"], "edge1 in chunk 5 (out of scope)", 5)
    hg.embeddings = np.array([[1.0, 0.0], [0.95, 0.05]], dtype=np.float32)

    query = np.array([1.0, 0.0], dtype=np.float32)
    activated = hg.diffuse(query, candidate_chunk_ids={0}, seed_k=1, hops=2)
    assert activated == [0]   # edge 1 is fenced out by the candidate set

