# 📖 Master Glossary — Every Term, One Place

> Every technical word used in this project, defined simply. If a term ever trips you up, it's
> here. (Grouped by theme, then roughly in learning order.)

## LLM & language basics
- **LLM (Large Language Model)** — a big neural network (Claude, GPT, Llama) that predicts the next text; can answer, write, summarize.
- **Token** — a chunk of text (~¾ of a word) that LLMs actually read; everything is measured and priced in tokens.
- **Context window** — the max tokens a model can read at once. Why we can't paste a whole codebase in.
- **Prompt** — the full text sent to the LLM (rules + context + question).
- **System prompt** — the instructions that set the model's behavior/rules.
- **Temperature** — randomness knob; low = factual, high = creative. We use low.
- **Hallucination** — the model confidently stating something false; RAG + citations fight it.

## RAG & retrieval
- **RAG (Retrieval-Augmented Generation)** — retrieve relevant chunks → put in the prompt → LLM answers from them. The whole project.
- **Retrieve / Augment / Generate** — the three RAG steps.
- **Grounding** — forcing the model to answer only from provided sources.
- **Citation** — a reference `[n]` linking a claim to its exact source; makes answers verifiable.
- **Fine-tuning** — the *other* way to give a model knowledge: retrain its weights. Contrast with RAG (see foundations doc).

## Chunking
- **Chunk** — one searchable piece of a file (a function, a method, a doc section).
- **Chunking** — splitting files into chunks.
- **AST (Abstract Syntax Tree)** — a tree describing code's structure; how we split code on logical boundaries.
- **Breadcrumb** — a doc section's chain of parent headings (its location/context).
- **Fallback chunker** — the safe default (overlapping line windows) for files we can't parse smartly.
- **Metadata** — extra info attached to a chunk (file, kind, line numbers) — used for filtering and citations.

## Embeddings & vectors
- **Embedding / vector** — a list of numbers representing a text's meaning.
- **Embedding model** — the neural network that makes embeddings (we use bge-small).
- **Dimension** — how many numbers per vector (384 for ours).
- **Semantic / dense search** — searching by meaning (via embeddings).
- **Cosine similarity** — the math for "how similar are two vectors" (1 = identical).
- **Normalization** — scaling vectors to length 1 so similarity is a clean, fast comparison.
- **Asymmetric embedding** — embedding queries and documents slightly differently (bge wants a query prefix).

## Vector database
- **Vector database** — stores vectors and finds the nearest ones fast (we use ChromaDB).
- **ANN (Approximate Nearest Neighbour)** — fast "find nearby vectors" search (trades a hair of accuracy for big speed).
- **HNSW** — the specific graph-based index algorithm ChromaDB uses for ANN.
- **Distance vs similarity** — Chroma returns a distance (0 = identical); we show `1 − distance` as similarity.

## Hybrid search
- **Lexical / keyword / sparse search** — matching exact words (BM25).
- **BM25** — the classic keyword-ranking algorithm; great for exact identifiers.
- **Dense vs sparse** — meaning-vectors (all non-zero) vs word-count vectors (mostly zeros).
- **Hybrid search** — combining semantic + keyword retrieval.
- **RRF (Reciprocal Rank Fusion)** — merging two ranked lists using ranks (not raw scores); scale-free.

## Reranking
- **Reranking** — reordering retrieved candidates with a more accurate model.
- **Bi-encoder** — encodes query and doc *separately* (fast; used in retrieval).
- **Cross-encoder** — encodes query and doc *together* (accurate; used in reranking).
- **Two-stage retrieval** — cheap retrieval to narrow, then expensive rerank to sharpen.
- **Candidate pool** — the larger set retrieved so the reranker has options to choose from.

## Agentic
- **Agent / agentic** — a program that decides its own steps during a task.
- **Query decomposition** — splitting a complex question into focused sub-questions.
- **Multi-hop question** — one needing evidence from several places.
- **Self-critique / grounding gate** — the system grading whether its own answer is supported.
- **Retry cap** — a hard limit so an agent loop can't run forever.

## Evaluation
- **Golden set / test set** — fixed questions with known-good answers; the "answer key".
- **Evaluation harness** — the script that runs the golden set and scores the system.
- **Context recall** — did retrieval fetch the chunk containing the answer?
- **Answer correctness** — does the answer contain the expected fact?
- **Faithfulness** — is the answer supported by its sources (no hallucination)?
- **LLM-as-judge** — using an LLM to grade outputs (runs offline).
- **Regression** — a change that makes a metric worse; the harness catches these.

## Engineering patterns
- **Interface / adapter** — a swappable contract (`LLMClient`, `Retriever`); swap implementations without touching callers.
- **Program to an interface, not an implementation** — depend on abstractions so pieces are replaceable.
- **Factory** — a function that builds the right object from a config (`build_pipeline`).
- **Metric-driven development** — validating every change against measured numbers.

## Tools
- **Python** — the language. **Streamlit** — turns a Python script into a web app.
- **sentence-transformers** — library for embeddings + cross-encoders.
- **ChromaDB** — the embedded vector database. **rank-bm25** — the keyword search library.
- **Groq** — the free, fast LLM API we use. **python-dotenv** — loads secrets from `.env`.
- **venv** — an isolated Python environment so libraries don't clash.
