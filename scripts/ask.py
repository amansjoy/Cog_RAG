"""Ask a question against the prebuilt Cog-RAG index.

Usage from terminal:
    python scripts/ask.py
    python scripts/ask.py --question "..." --mode cog
    python scripts/ask.py --mode both

In PyCharm, create a Run Configuration with:
    Script path:        scripts/ask.py
    Parameters:         --question "What links sleep disruption to atrial fibrillation?"
    Working directory:  project root
"""
import argparse
import sys
from pathlib import Path

# Make the project root importable when this script is launched directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cog_rag.config import make_client
from cog_rag.index import CogRagIndex
from cog_rag.retrieval import cog_rag_answer, vanilla_rag_answer

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "data" / "cog_rag_index.pkl"

DEFAULT_QUESTION = (
    "How does chronic stress affect cardiovascular health in middle-aged adults? "
    "Trace the main cause-effect chains."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--mode", choices=["cog", "vanilla", "both"], default="both")
    parser.add_argument("--top-k-themes", type=int, default=4)
    parser.add_argument("--seed-k", type=int, default=6)
    parser.add_argument("--hops", type=int, default=2)
    args = parser.parse_args()

    if not INDEX_PATH.exists():
        raise SystemExit(f"Index not found at {INDEX_PATH}. Run scripts/build_index.py first.")

    client = make_client()
    index = CogRagIndex.load(str(INDEX_PATH))
    # print(f"[index] {len(index.chunks)} chunks, "
    #       f"{len(index.theme_hg.themes)} themes, "
    #       f"{len(index.entity_hg.edges)} hyperedges\n")

    if args.mode in ("vanilla", "both"):
        print("=" * 30 + " VANILLA RAG " + "=" * 30)
        out = vanilla_rag_answer(client, index, args.question)
        print(out["answer"])
        print()

    if args.mode in ("cog", "both"):
        print("=" * 32 + " COG-RAG " + "=" * 32)
        out = cog_rag_answer(
            client, index, args.question,
            top_k_themes=args.top_k_themes,
            seed_k=args.seed_k,
            hops=args.hops,
        )
        #print("Themes activated:", out["themes"])
        #print("\nHigh-order relations activated:")
        # for ents, rel, cid in out["relations"]:
        #     print(f"  ({', '.join(ents)}) :: {rel}  [chunk {cid}]")
        print("\nAnswer:\n")
        print(out["answer"])


if __name__ == "__main__":
    main()
