# Phase 1 — Chunking (Splitting Files Into Meaningful Pieces)

## 🎯 What this phase does
Takes a folder of code and documents and splits every file into small, self-contained
**chunks** — a whole function, a whole class method, a documentation section. These chunks
are the raw material the whole system searches over later.

---

## 🧠 The big idea first: why "chunking" even matters

An LLM can only read a limited amount of text at once (its "context window"). You can't
paste a whole codebase in. So RAG works by searching for the *few most relevant pieces* and
sending only those to the LLM.

But that raises the question: **what is a "piece"?** How do you cut a file up?

- ❌ **The naive way:** cut every 500 characters. This slices functions in half. A search
  then returns "the bottom half of a function" — meaningless to both the search engine and
  the LLM.
- ✅ **Our way:** cut on *logical boundaries* — a whole function, a whole method, a whole
  doc section. Every chunk is a complete, understandable unit.

**This single decision — smart chunking — is one of the biggest levers on RAG quality.**
Garbage chunks = garbage answers, no matter how fancy the rest of the system is.

---

## 🔩 What we built (the files)

```
src/chunking/
  base.py            → the Chunk data shape (id, content, metadata)
  python_chunker.py  → splits Python code using its AST
  markdown_chunker.py→ splits docs using their headings
  dispatcher.py      → picks the right splitter per file + a safety fallback
ingest.py            → the command you run: folder → chunks.json + a summary
```

## 1. The `Chunk` — our unit of knowledge

Every chunk has three parts:
- **`chunk_id`** — a stable, readable name like `example.py::BankAccount.deposit`.
- **`content`** — the actual text (the function's code, the doc section's text).
- **`metadata`** — everything else we know: file, language, kind, line numbers, etc.

**Why a stable id?** Because later, if one function changes, we want to update *just that
chunk* instead of rebuilding everything. A predictable id makes that possible.

## 2. AST chunking for Python (the star of this phase)

**What is an AST?** *Abstract Syntax Tree.* When Python reads your code, it first turns it
into a tree that describes its structure: "this is a function named `deposit`, it lives
inside class `BankAccount`, it spans lines 18–22." Python gives us this for free via the
built-in `ast` module — the *same* parser the interpreter uses. **No external library.**

We walk that tree and emit one chunk for:
- the module's top docstring,
- each top-level function,
- each class (a small "overview" chunk = its signature + docstring),
- each method inside a class.

**Why split a class into overview + methods** instead of one big chunk? A large class as a
single chunk would be huge and unfocused — it would match *everything* vaguely and *nothing*
precisely. Splitting keeps each chunk sharp.

**Proof it works** (real output from our sample):
```
[module_doc    ] example.py::<module docstring>
[function      ] example.py::area_of_circle
[class_overview] example.py::BankAccount
[method        ] example.py::BankAccount.__init__
[method        ] example.py::BankAccount.deposit
[method        ] example.py::BankAccount.withdraw
```
Every chunk is a complete, nameable thing. No half-functions.

## 3. Structural chunking for Markdown/docs

For documents, the natural boundary is the **heading**. We split into one section per
heading, and — the clever bit — we remember each section's **breadcrumb** (its chain of
parent headings):
```
example.md::Payments Service > Configuration > API Keys
```
**Why the breadcrumb matters:** a section titled "API Keys" is ambiguous alone. But
"Payments Service > Configuration > API Keys" tells the LLM *exactly* where it sits. That
context dramatically improves answer quality. We store it in metadata.

## 4. The dispatcher + the safety net

`dispatcher.py` looks at each file's extension and routes it:
- `.py` → AST chunker
- `.md` → heading chunker
- anything else → a **fallback** that cuts into overlapping 60-line windows.

**Why a fallback?** We should *never silently drop a file* just because we don't have a
fancy parser for its language yet. A rough chunk beats no chunk. (Later we upgrade the
fallback to `tree-sitter` for real AST parsing across many languages.) The Python chunker
also falls back automatically if a file has a syntax error — robustness over perfection.

---

## 🔑 Words you must know (this phase)

- **Chunk** — one searchable piece of a file.
- **Chunking** — the process of splitting files into chunks.
- **AST (Abstract Syntax Tree)** — a tree describing code's structure; how we split code intelligently.
- **Metadata** — extra info attached to a chunk (file, kind, line numbers).
- **Breadcrumb** — a doc section's chain of parent headings (its "location").
- **Fallback** — the safe default splitter for files we can't parse smartly.
- **Context window** — the max amount of text an LLM can read at once (the reason we chunk).

---

## 🛡️ Interview defense (say these out loud)

> *"Why not just split text every N characters?"*
> "Fixed-size splitting cuts functions and ideas in half, so retrieval returns fragments.
> I chunk on **logical boundaries** — I parse Python's **AST** to split on whole functions,
> classes, and methods, and I split Markdown on its heading structure while keeping a
> breadcrumb of parent headings. Each chunk is a complete, self-contained unit, which is one
> of the biggest levers on retrieval quality."

> *"How do you handle a language you don't have a parser for?"*
> "A safety-net fallback chunker splits any file into overlapping line windows, so I never
> drop a file. The AST path also falls back automatically on a syntax error. Later I'd swap
> the fallback for tree-sitter to get real structural parsing across languages."

> *"Why store metadata and breadcrumbs?"*
> "Retrieval quality and citations. Line numbers and file paths let me cite exact sources;
> the heading breadcrumb disambiguates a section so the LLM knows where it came from."

**Keywords to drop:** *logical/structural chunking, Abstract Syntax Tree, semantic
boundaries, metadata for citations, graceful fallback, retrieval quality.*

---

## ✅ What you can now say you built
1. A chunker that splits Python by **AST** into functions, classes, and methods.
2. A Markdown chunker that splits by headings and records breadcrumb paths.
3. A dispatcher with a **fallback** so no file is ever dropped.
4. A CLI (`ingest.py`) that turns any folder into inspectable `chunks.json`.

➡️ Next: [phase-02-embeddings-search.md](phase-02-embeddings-search.md) — turning chunks into
searchable meaning-vectors.
