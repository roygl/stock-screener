# DEPLOY.md — Hosting & Deployment

The screener is a **Streamlit** app deployed to **[Streamlit Community Cloud](https://share.streamlit.io)**
(free, GitHub-connected). It auto-builds from this repo's `main` branch.

> **Why not Cloudflare / a Worker?** Streamlit is a long-running Tornado + WebSocket
> server with native deps (`pandas`/`numpy`/`pyarrow`/`yfinance`), so it can't run on
> Cloudflare Workers or Pages. The only Cloudflare path is **Containers** (Workers
> **Paid** plan, ~$5/mo). Streamlit Community Cloud hosts this exact app for free with
> no Dockerfile. See `DECISIONS.md` (2026-06-20). RAM fallback: Hugging Face Spaces
> (free, ~16 GB).

## What Streamlit Cloud needs (all already true in this repo)

- `app.py` at the repo root — the entry point.
- `requirements.txt` at the repo root — `>=` floors that resolve cleanly on Cloud's Python.
- Pure-pip deps ⇒ **no `packages.txt`** (no apt/system libraries).
- App is **offline-first** ⇒ runs with **zero secrets** (the NL agent degrades to the
  rule-based parser without an API key).

## First deploy (one-time, web flow — no CLI)

1. Go to **https://share.streamlit.io** and sign in with GitHub; authorize access to
   `roygl/stock-screener`.
2. **Create app** → deploy from an existing repo.
3. Set:
   - **Repository:** `roygl/stock-screener`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. **Advanced settings → Python version: 3.12** (safe for the numpy/pandas/pyarrow set).
5. **Deploy.** First build compiles wheels (~2–4 min); you get a public
   `https://<name>.streamlit.app` URL.

After this, every push to `main` redeploys automatically.

## Optional — enable the LLM agent

Without a key the agent uses the offline rule-based parser. To enable an LLM backend,
open **app → Settings → Secrets** and add **top-level** keys (Streamlit exports only
root-level secrets as environment variables, which the SDKs read — keys nested under a
`[section]` are **not** exported):

```toml
ANTHROPIC_API_KEY = "sk-ant-..."

# To use a different backend instead of the Anthropic default:
# SCREENER_AGENT_PROVIDER = "openai"   # openai | xai | gemini | mistral | ollama
# OPENAI_API_KEY = "sk-..."
# SCREENER_AGENT_MODEL = "gpt-4o"      # optional per-provider model override
```

Provider → env-var key (see `PROVIDERS` in `screener/agent.py`):
`anthropic`→`ANTHROPIC_API_KEY`, `openai`→`OPENAI_API_KEY`, `xai`→`XAI_API_KEY`,
`gemini`→`GEMINI_API_KEY`, `mistral`→`MISTRAL_API_KEY`. (`ollama` is keyless/local and
not reachable from Streamlit Cloud.)

## Operational notes

- **Idle sleep:** free apps sleep without traffic and cold-start (~30s) on the next visit.
- **yfinance from cloud IPs:** Yahoo occasionally rate-limits datacenter egress, so a scan
  may return sparse — re-run; the on-disk parquet cache (`.cache/`, ephemeral on Cloud)
  absorbs most repeat pulls.
- **~1 GB RAM tier:** if a full-universe scan OOMs, scan fewer names or move to Hugging
  Face Spaces (same repo, no code changes).
