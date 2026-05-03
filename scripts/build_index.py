"""Build the Cog-RAG index by running theme + entity extraction over the PDF.

This is the slow step: ~1 LLM call per chunk for extraction, plus embedding
calls for chunks/themes/hyperedges. The result is pickled to data/cog_rag_index.pkl
and reused by ask.py.

PyCharm: create a Run Configuration with this file as the script and the
project root as the working directory.
"""
import sys
from pathlib import Path

# Make the project root importable when this script is launched directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cog_rag.chunking import read_pdf
from cog_rag.config import make_client
from cog_rag.index import build_index

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "data" / "chronic_stress_cvd_review.pdf"
INDEX_PATH = ROOT / "data" / "cog_rag_index.pkl"


def main() -> None:
    if not PDF_PATH.exists():
        raise SystemExit(f"PDF not found at {PDF_PATH}. Run scripts/build_pdf.py first.")
    client = make_client()
    text = read_pdf(str(PDF_PATH))
    index = build_index(client, text)
    index.save(str(INDEX_PATH))
    print(f"\n[done] saved index to {INDEX_PATH}")
    print(f"       chunks={len(index.chunks)} "
          f"themes={len(index.theme_hg.themes)} "
          f"hyperedges={len(index.entity_hg.edges)}")


if __name__ == "__main__":
    main()
