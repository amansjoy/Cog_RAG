# Cog-RAG, modular project

A PyCharm-friendly implementation of **Cog-RAG**
(*Cognitive-Inspired Dual-Hypergraph with Theme Alignment RAG*,
Hu et al., 2025, arXiv:2511.13201) over a 45-page synthesised medical
review on chronic stress and cardiovascular health.

The previous single-file version is kept for readability; this one is
split into a real Python package so PyCharm gives you proper navigation,
refactoring, run configurations, and tests.

---

## Project layout

```
cog_rag_demo/
├── README.md
├── requirements.txt
├── .env.example                            # copy to .env and fill in
├── .gitignore
│
├── data/                                   # generated artefacts go here
│   ├── chronic_stress_cvd_review.pdf       # produced by scripts/build_pdf.py
│   └── cog_rag_index.pkl                   # produced by scripts/build_index.py
│
├── content/                                # the medical text as Python data
│   ├── __init__.py                         #   exposes ALL_CONTENT
│   ├── part1.py … part5.py                 #   chapter content as (level, text) tuples
│   └── pdf_builder.py                      #   reportlab assembly
│
├── cog_rag/                                # ★ the actual package you import
│   ├── __init__.py                         #   public API
│   ├── config.py                           #   env, Settings, AzureOpenAI factory
│   ├── llm.py                              #   embed(), chat(), cosine helpers
│   ├── chunking.py                         #   read_pdf(), chunk_text()
│   ├── extraction.py                       #   theme + high-order entity extractor
│   ├── hypergraphs.py                      #   ThemeHypergraph, EntityHypergraph
│   ├── index.py                            #   CogRagIndex, build_index, save/load
│   └── retrieval.py                        #   cog_rag_answer, vanilla_rag_answer
│
├── scripts/                                # ★ entry points → PyCharm Run Configs
│   ├── build_pdf.py
│   ├── build_index.py
│   └── ask.py
│
└── tests/
    └── test_smoke.py                       # pure-Python tests (no Azure needed)
```

Each module does one thing. Read top-to-bottom in this order to follow the
data flow: `chunking → extraction → hypergraphs → index → retrieval`.

---

## What was redefined

**Nothing was redefined logically.** Same functions, same algorithms, same
prompts as the single-file version. The only changes are mechanical:

| Single-file version              | Modular version                            |
|----------------------------------|--------------------------------------------|
| `make_client()`                  | `cog_rag.config.make_client`               |
| `embed`, `chat`, `cosine_matrix` | `cog_rag.llm`                              |
| `read_pdf`, `chunk_text`         | `cog_rag.chunking`                         |
| `extract`, `EXTRACTION_PROMPT`   | `cog_rag.extraction`                       |
| `ThemeHypergraph`, `EntityHypergraph` | `cog_rag.hypergraphs`                 |
| `CogRagIndex`, `build_index`     | `cog_rag.index`                            |
| `cog_rag_answer`, `vanilla_rag_answer` | `cog_rag.retrieval`                  |
| `main()` argparse block          | split into 3 scripts in `scripts/`         |

Two small additions you didn't have before, both worth knowing about:

1. **`Settings` dataclass** in `config.py` — central place where env vars
   are read. If you want to override deployment names from a test, build
   a `Settings` object instead of monkey-patching `os.environ`.

2. **`get_settings()`** — module-level singleton so `llm.py` doesn't
   re-read env vars on every embedding call.

---

## PyCharm setup

### 1. Open the project

`File → Open` and select the `cog_rag_demo` folder. **Do not** open the
parent folder; PyCharm needs the project root to be the directory that
contains `cog_rag/`, `content/`, and `scripts/`.

### 2. Configure the interpreter

`File → Settings → Project → Python Interpreter → Add Interpreter →
Add Local Interpreter → Virtualenv Environment → New`.

Then in the terminal at the project root:
```bash
pip install -r requirements.txt
```

### 3. Mark sources root (important)

Right-click the project root in the Project pane → **Mark Directory as →
Sources Root**.

This is what makes `from cog_rag.index import …` resolve in both your IDE
and at runtime when scripts in `scripts/` are executed. Without it, the
imports underline red.

### 4. Create the .env file

```bash
cp .env.example .env
# edit .env and put in your Azure values
```

`cog_rag/config.py` calls `load_dotenv()` automatically, so PyCharm will
pick up the values whenever you run any script.

If you prefer the PyCharm-native way: open Run → Edit Configurations,
and put the variables into the **Environment variables** field. Either
approach works; the dotenv path is just less fiddly.

### 5. Run configurations

Create three Run Configurations (Run → Edit Configurations → `+` → Python):

| Name             | Script path              | Working directory  | Parameters                                |
|------------------|--------------------------|--------------------|-------------------------------------------|
| Build PDF        | `scripts/build_pdf.py`   | `<project root>`   | (none)                                    |
| Build index      | `scripts/build_index.py` | `<project root>`   | (none)                                    |
| Ask (default)    | `scripts/ask.py`         | `<project root>`   | (none)                                    |
| Ask (custom)     | `scripts/ask.py`         | `<project root>`   | `--question "your question" --mode cog`   |

The "working directory must be project root" point is the one thing that
catches everyone the first time. If you launch with the working directory
set to `scripts/`, the relative `data/` paths break.

### 6. Run order

```
Build PDF       →   creates data/chronic_stress_cvd_review.pdf
Build index     →   creates data/cog_rag_index.pkl   (slow: ~1 LLM call per chunk)
Ask             →   queries the index   (fast)
```

You only run **Build index** once. Every subsequent question is just **Ask**.

---

## Cog-RAG, in two paragraphs

Vanilla RAG flattens documents into chunks and grabs top-k by cosine.
That misses two things: global structure (related ideas scattered across
chapters) and high-order facts (cause-effect chains involving 3+
entities, which a graph can only express as broken pairwise edges).

Cog-RAG fixes both by building **two hypergraphs** at index time and
running **two-stage retrieval** at query time:

- **Theme hypergraph**: each hyperedge is a theme that links *every*
  chunk that touches it, regardless of where in the document. Stage 1
  embeds the query, picks the top themes, and the union of their chunks
  is the *candidate set*.
- **Entity hypergraph**: each hyperedge is a multi-entity cause-effect
  relation (`{cortisol, sodium retention, vasoconstriction, hypertension}`
  → "cortisol amplifies vasoconstrictor sensitivity and promotes sodium
  reabsorption, raising arterial pressure"). Stage 2 picks seed
  hyperedges by similarity *inside the candidate set*, then diffuses
  through shared entities for `h` hops to pull in connected mechanisms.

The result is evidence that is globally organised by theme and locally
detailed by multi-entity relations — exactly what the answer prompt then
gets to work with.

For the full conceptual walkthrough, paper-to-code mapping, and
discussion of caveats, read the original explanation
(it is unchanged; only the file paths shifted).

---

## Run the tests

```bash
pytest tests/
```

These don't need Azure credentials. They exercise chunking and the
hypergraph diffusion logic directly. If diffusion is what you want to
understand, `tests/test_smoke.py::test_entity_hypergraph_diffusion_finds_shared_entity`
is a tiny worked example you can step through with the PyCharm debugger.

---

## Where to extend

- **Different document**: drop your PDF into `data/`, point `scripts/build_index.py`
  at it, and rebuild. Nothing else needs to change.
- **Different domain prompts**: edit `cog_rag/extraction.py::EXTRACTION_PROMPT`
  and `cog_rag/retrieval.py::ANSWER_PROMPT`. These are the only domain-specific bits.
- **Different LLM provider**: swap `cog_rag/config.py::make_client` and
  `cog_rag/llm.py::embed/chat`. The rest of the package never touches the API.
- **Theme deduplication**: cluster near-duplicate themes before
  `theme_hg.finalize()` to reduce noise (mentioned as a known limitation).
