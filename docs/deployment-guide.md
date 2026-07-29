# 🚀 Deployment Guide — Follow This Tomorrow (Beginner-Friendly)

> Goal: turn your locally-running project into a **public website with a link** you can put on
> your resume. This guide explains the *concepts* first (so you understand what you're doing),
> then gives exact, copy-paste steps. Budget ~45–60 minutes. Nothing here runs today — it's your
> plan for tomorrow.

---

## Part 1 — Concepts you need first (10 min read)

**What is "deployment"?**
Right now your app runs only on *your* laptop. Deployment means putting it on a computer on the
internet (a "server") so anyone with the link can use it. You don't rent a server yourself — you
use a free hosting platform that runs it for you.

**Which platform?** → **Streamlit Community Cloud** (free).
Your app's UI is built with Streamlit, and Streamlit has its own free hosting made exactly for
this. It reads your code from GitHub and runs it. Easiest possible path for a beginner.

**The 4 things a host needs from you (and the concept behind each):**

1. **Your code on GitHub.** The host pulls your code from a GitHub repository (a cloud copy of
   your project). So step one is putting your project on GitHub.
2. **A `requirements.txt`.** The host has a bare Python — it must install your libraries. This
   file is the shopping list of exact packages to install.
3. **Your secret (the Groq API key) — kept OUT of the code.** You must NEVER put your API key in
   GitHub (anyone could steal it). Instead you paste it into the host's private **"Secrets"** box.
   The app reads it from there at runtime. (This is *the* most important security concept in
   deployment.)
4. **The search index.** Your app needs the `chunks.json` + `.chroma` index to answer questions.
   We'll include a small pre-built index in the repo so the live app has data to search.

**One more concept — "ephemeral filesystem":** hosted apps can be restarted anytime, and any
files they *created* while running may be wiped. So anything the app *needs* must come from the
repo (committed), not be assumed to already exist on the server. That's why we commit the index.

---

## Part 2 — The steps (do these tomorrow, in order)

### ✅ Step 0 — Pre-flight (5 min)
- **Rotate your Groq key** if you haven't (you shared it in chat). Groq console → delete old →
  create new → keep it handy. You'll paste it into Streamlit's Secrets, not into code.
- Confirm the app runs locally one more time:
  ```bash
  cd ~/Desktop/PROJECTS/code-intel-engine
  ./.venv/bin/streamlit run app.py
  ```
  Ask a question, confirm you get a cited answer, then Ctrl+C to stop.

### ✅ Step 1 — Finalize `requirements.txt` (2 min)
Replace the contents of `requirements.txt` with this exact list (the host installs these):
```
streamlit
sentence-transformers
chromadb
rank-bm25
groq
python-dotenv
```
*Why:* these are every library your app imports. If one is missing, the host's build fails.

### ✅ Step 2 — Bundle the search index into the repo (5 min)
The live app needs data to search. We'll ship a small pre-built index.

1. Build a fresh index locally from the sample (or point at any folder you want to demo):
   ```bash
   ./.venv/bin/python ingest.py sample_input chunks.json
   ./.venv/bin/python build_index.py chunks.json
   ```
2. Tell git to INCLUDE the index (it's currently ignored). Open `.gitignore` and **delete these
   two lines**:
   ```
   chunks.json
   .chroma/
   ```
   *Why:* normally you don't commit generated data, but for a self-contained demo we want the
   index to travel with the code so the live app has something to answer about.

### ✅ Step 3 — Bridge the secret into the app (3 min)
Streamlit stores secrets in `st.secrets`, but your Groq client reads from `os.environ`. Add this
tiny bridge at the **very top** of `app.py` (right after `import streamlit as st`):
```python
import os
# On Streamlit Cloud the key comes from the Secrets box; copy it into the environment
# so the existing Groq client (which reads os.environ) finds it. Locally, .env handles this.
if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
```
*Why:* this is the glue between "the host's secret box" and "the code that needs the key" —
without touching how the rest of the app works.

### ✅ Step 4 — Put the project on GitHub (10 min)
1. Create a **free GitHub account** if you don't have one (github.com).
2. In your project folder, initialize git and commit:
   ```bash
   cd ~/Desktop/PROJECTS/code-intel-engine
   git init
   git add .
   git status          # LOOK: confirm .env is NOT listed (it must stay secret!)
   git commit -m "Code-Intel Engine: RAG codebase intelligence tool"
   ```
   ⚠️ **Critical check:** `git status` must NOT show `.env`. If it does, stop — your `.gitignore`
   should be excluding it. Never commit the key.
3. On github.com click **New repository** → name it `code-intel-engine` → **Public** → *Create*
   (don't add a README, you already have one).
4. Connect and push (GitHub shows you these exact lines after creating the repo):
   ```bash
   git remote add origin https://github.com/<your-username>/code-intel-engine.git
   git branch -M main
   git push -u origin main
   ```

### ✅ Step 5 — Deploy on Streamlit Community Cloud (10 min)
1. Go to **share.streamlit.io** → **Sign in with GitHub** → authorize it.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `<your-username>/code-intel-engine`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Advanced settings**:
   - **Python version:** 3.12
   - **Secrets:** paste this (TOML format), with your real key:
     ```toml
     GROQ_API_KEY = "gsk_...your rotated key..."
     ```
5. Click **Deploy**. It installs your requirements (takes a few minutes the first time — it's
   downloading PyTorch and your models) and starts the app.

### ✅ Step 6 — Test your live app (5 min)
- You'll get a public URL like `https://code-intel-engine.streamlit.app`.
- Open it, ask a question, confirm you get a cited answer. **That link is now on the internet.**
- Put it on your resume: *"Live demo: <url>"*.

---

## Part 3 — Troubleshooting (if something breaks)

| Symptom | Cause | Fix |
|---|---|---|
| Build fails: "No module named X" | `requirements.txt` missing a package | add it to `requirements.txt`, commit, push (auto-redeploys) |
| "GROQ_API_KEY is not set" on the live app | secret not saved | re-open app → **Settings → Secrets**, paste the TOML, save |
| App crashes / "Error / resource limits" | out of memory (free tier ~1 GB; PyTorch + reranker is heavy) | in `app.py` change the default retrieval strategy from `hybrid_rerank` to `dense` (skips loading the reranker model) |
| Live app says "index is empty" | the `.chroma` index wasn't committed | confirm you removed it from `.gitignore` (Step 2) and re-pushed |
| Accidentally committed `.env` | key exposed | **rotate the key immediately** in Groq console, then remove the file from git |

**Redeploy after any change:** just `git add . && git commit -m "fix" && git push` — Streamlit
auto-redeploys on every push. That's the whole update loop.

---

## Part 4 — What you learned (say this in interviews)

After doing this you can honestly say:
> *"I deployed it to Streamlit Community Cloud from a GitHub repo. I manage the API key as a
> platform secret rather than committing it, install dependencies via requirements.txt, and ship
> a pre-built index because the host filesystem is ephemeral. Pushing to main auto-redeploys."*

**Keywords:** *deployment, GitHub, CI/auto-deploy, environment secrets, requirements/dependency
management, ephemeral filesystem, resource limits.*

---

## Alternative platforms (for later, optional)
- **Hugging Face Spaces** — also free, supports Streamlit, sometimes more RAM. Same idea.
- **Docker + Render/Railway** — more control, more setup. Good "next level" once you're comfortable.
