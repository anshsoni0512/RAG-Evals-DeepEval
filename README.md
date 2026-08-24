# RAG Evals with DeepEval framework

A from-scratch **evaluation suite for a Retrieval-Augmented Generation (RAG) pipeline**, built to
measure retrieval quality, reranking quality, generation quality, and end-to-end pipeline quality.
It uses [DeepEval](https://github.com/confident-ai/deepeval) with an **LLM-as-a-judge** pattern,
powered by Groq-hosted open-weight models.

The corpus is a set of transcribed lecture sessions on LLM evaluation itself, so this repo is,
fittingly, a RAG system that answers questions about how to evaluate RAG systems, with a full
eval harness wrapped around every stage of the pipeline.

## Why this project

Most RAG tutorials stop at "it retrieves, it generates, ship it." This project instead treats
each stage of the pipeline as its own component with its own quality bar, and builds golden
datasets and metrics to test each one independently:

- Is the **retriever** finding the right chunks? (Contextual Precision / Recall)
- Does **reranking** actually improve the ordering over raw retrieval?
- Is the **generator** faithful to its context, and relevant to the question? (Faithfulness /
  Answer Relevancy)
- Does the **full pipeline**, end to end, hold up on all of the above at once?

## Architecture

```
data/*.vtt (lecture transcripts)
        │
        ▼
  chunk + embed  ────────────►  Pinecone (vector store)
  (RecursiveCharacterTextSplitter,
   chunk_size=1400, overlap=200)
        │
        ▼
┌──────────────────┐      ┌───────────────────────┐      ┌──────────────────────┐
│  Retriever        │ ──►  │  Reranker              │ ──►  │  Generator            │
│  (dense, top-k=5) │      │  (over-fetch k=10,     │      │  (Groq LLM, grounded  │
│  llama-text-embed │      │   cross-encoder rerank │      │   strictly in context)│
│  -v2 embeddings)  │      │   then top-k=5)        │      │                       │
└──────────────────┘      └───────────────────────┘      └──────────────────────┘
        │                          │                                │
        └──────────────────────────┴────────────────────────────────┘
                                    ▼
                            src/pipeline.py
                    (single entry point: query in, answer out)
```

## Evaluation suite

Every stage above has a matching eval script under `evals/`, each scored by an LLM judge
(`openai/gpt-oss-120b`, served through Groq's OpenAI-compatible endpoint and wrapped with
DeepEval's `LocalModel`) against a hand-written golden dataset.

| Script                       | Component tested                          | Metrics                                                                     | Golden data                            |
|-------------------------------|--------------------------------------------|-------------------------------------------------------------------------------|-------------------------------------------|
| `evals/eval_retriever.py`    | Dense retriever                            | `ContextualPrecisionMetric`, `ContextualRecallMetric`                        | `goldendata/golden_retriever.json`      |
| `evals/eval_reranker.py`     | Cross-encoder reranker                     | `ContextualPrecisionMetric`, `ContextualRecallMetric`                        | `goldendata/golden_retriever.json`      |
| `evals/eval_generator.py`    | Answer generation                          | `FaithfulnessMetric`, `AnswerRelevancyMetric`                                | `goldendata/faithfulness_data.json`     |
| `evals/eval_pipeline.py`     | Full retrieve, rerank, generate pipeline   | `FaithfulnessMetric`, `AnswerRelevancyMetric`, `ContextualRelevancyMetric`   | `goldendata/faithfulness_data.json`     |
| `evals/eval_application.py`  | Full pipeline, judged against an ideal answer (reference-based, application-level quality) | `GEval` — custom `Correctness`, `Completeness`, `Style` rubrics | `goldendata/correctness_data.json`      |

**What each metric actually checks:**

- **Contextual Precision**: are the relevant chunks ranked above the irrelevant ones in the
  retrieved context?
- **Contextual Recall**: does the retrieved context contain everything needed to produce the
  ideal answer?
- **Contextual Relevancy**: of what was retrieved, how much of it is actually relevant to the
  query (signal-to-noise of retrieval)?
- **Faithfulness**: does the generated answer only claim things supported by the retrieved
  context (no hallucination)?
- **Answer Relevancy**: does the generated answer actually address the question that was asked?
- **Correctness (G-Eval)**: do the answer's factual claims avoid contradicting the golden
  `ideal_answer`? Judges truth only — brevity, omissions, and extra correct detail are never
  penalized.
- **Completeness (G-Eval)**: how many of the key points in the golden `ideal_answer` does the
  answer actually cover? Judges coverage only, independent of whether those points are phrased
  correctly.
- **Style (G-Eval)**: does the answer read in an intuitive, conversational "explain it out loud"
  teaching voice rather than a dry, bureaucratic, or bare-bullet-list tone?

## Components

- **`src/retriver.py`**: loads the `.vtt` transcripts, strips WEBVTT timestamps and metadata,
  chunks the text, embeds it with Pinecone's `llama-text-embed-v2`, and upserts it into a
  Pinecone index (`deepevals`). Skips re-embedding if the index is already populated.
- **`src/reranker.py`**: wraps the dense retriever with a `cross-encoder/ms-marco-MiniLM-L-6-v2`
  reranker. It over-fetches `k=10` candidates from the bi-encoder, then reranks and keeps the top
  `5` most relevant.
- **`src/generator.py`**: a strict, context-grounded answer generator (Groq `openai/gpt-oss-20b`)
  that refuses to answer when the provided context doesn't cover the question.
- **`src/pipeline.py`**: connects retrieval, reranking, and generation into a single
  `Pipeline.invoking(query)` call, returning the query, the retrieved context, and the final
  answer together.

## Tech stack

- **Orchestration:** LangChain (`langchain-core`, `langchain-groq`, `langchain-pinecone`,
  `langchain-text-splitters`)
- **Vector store:** Pinecone
- **Reranking:** `sentence-transformers` (CrossEncoder)
- **LLM inference:** Groq (OpenAI-compatible endpoint). `openai/gpt-oss-20b` for generation,
  `openai/gpt-oss-120b` as the judge for the retrieval/generation/pipeline evals, `gpt-4o-mini`
  (OpenAI, via DeepEval's built-in `GEval`) as the judge for the application-level eval.
- **Evaluation framework:** DeepEval
- **Tooling:** `uv` for dependency and environment management

## Setup

```bash
uv sync
```

Create a `.env` file with:

```
PINECONE_API_KEY=...
GROQ_API_KEY=...
OPENAI_API_KEY=...   # only needed for evals/eval_application.py (GEval judge = gpt-4o-mini)
```

Run the pipeline directly:

```bash
uv run python -m src.pipeline
```

Run any of the evals (needs `PYTHONUTF8=1` on Windows for clean output encoding):

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run python -m evals.eval_retriever
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run python -m evals.eval_reranker
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run python -m evals.eval_generator
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run python -m evals.eval_pipeline
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 uv run python -m evals.eval_application
```

## Engineering notes

A few non-obvious decisions worth explaining:

- **LLM-as-judge over Groq, not OpenAI.** DeepEval has no built-in Groq wrapper, so the judge is
  built by pointing DeepEval's generic `LocalModel` at Groq's OpenAI-compatible endpoint
  (`base_url="https://api.groq.com/openai/v1"`). This keeps the whole stack on one free-tier
  provider.
- **Rate-limit-aware evaluation.** Groq's free tier caps tokens per minute and tokens per day.
  Eval runs use `AsyncConfig(max_concurrent=1, throttle_value=16)` to stay under the per-minute
  cap during larger batches.
- **Caching disabled on purpose.** DeepEval's on-disk test-run cache uses shared file locks that
  need `pywin32` on Windows. Instead of adding that dependency, caching is explicitly turned off
  (`CacheConfig(write_cache=False, use_cache=False)`) so runs are always fresh and don't silently
  fail on a lock error.
- **Model deprecation handling.** Groq retired `llama-3.1-8b-instant` partway through this
  project. The generator was migrated to `openai/gpt-oss-20b` after checking the currently
  available models directly through the Groq SDK's `models.list()`, instead of guessing from
  docs that may be out of date.
- **Reference-based grading for the application eval.** Unlike the other evals, which check
  faithfulness/relevancy with no "correct answer" to compare against, `eval_application.py` grades
  against a golden `ideal_answer` per question. That needed three separate, narrowly-scoped
  `GEval` rubrics (Correctness, Completeness, Style) so an answer that's short-but-accurate isn't
  docked for incompleteness, and vice versa — a single blended metric would conflate those failure
  modes.

## Roadmap

- [x] Retriever eval (Contextual Precision / Recall)
- [x] Reranker eval (Contextual Precision / Recall, after reranking)
- [x] Generator eval (Faithfulness / Answer Relevancy)
- [x] Full pipeline eval (Faithfulness / Answer Relevancy / Contextual Relevancy on the pipeline's
      actual retrieved context and answer)
- [x] Application-level eval against golden ideal answers (Correctness / Completeness / Style via
      `GEval`)
- [ ] Migrate `eval_retriever.py`'s judge model off the retired `llama-3.1-8b-instant`
- [ ] CI-driven eval runs on pull requests, gated on metric thresholds
