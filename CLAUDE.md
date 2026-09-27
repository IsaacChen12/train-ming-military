# CLAUDE.md

Project-specific instructions for Claude Code when working in this repo.

## Environment notes

- This machine has no WSL2 and no Docker installed. The plain RAG pipeline (Stage 1+2) doesn't need either: `chromadb.PersistentClient` is a local embedded store, and Ollama is already installed natively on Windows (v0.34.4, with `qwen3.5:2b/4b/9b` pulled).
- System Python (Anaconda) is 3.14, which is incompatible with the old pinned versions in `00_setup/requirements.txt`. Use `py -3.11` instead. A project venv already exists at `.venv/` (Python 3.11) with the minimal deps needed for the data pipeline (`python-docx`, `tqdm`, `loguru`, `pymupdf`, `openai`) — not the full training/OCR stack from `requirements.txt`.
- Heavier LLM inference is meant to run on another host on the same LAN — this machine isn't powerful enough for generation (confirmed too slow with local `qwen3.5:9b`). The host's address is noted in `../RAGMing_可能用到的操作与密钥.txt` (one level up, outside this repo). That file also contains real credentials (GitHub password/token, Open-WebUI account password) — it's fine to read it for connection info, but never repeat the secrets in chat or send them to any external service.

## Documentation conventions

- `README.md` and `RUNBOOK.md` are bilingual: each Chinese paragraph is immediately followed by its English translation, interleaved within the same file (not separate `.en.md` files). Code blocks stay as a single version with bilingual inline comments. Keep any new content in this same format.
- `WORKLOG.md` (project root) is a running, bilingual log of actual hands-on work — commands run, code changed, bugs found/fixed, connectivity checks, etc. It is distinct from the "调整记录 / Changelog" section inside `RUNBOOK.md`, which is reserved specifically for model/parameter selection decisions, not general work.

## Maintenance rule

Whenever material is added to `data/raw/` and the extract → clean → chunk pipeline is re-run (changing `data/chunks/`), update all three of:

1. The corpus-statistics table in `README.md` ("当前语料统计 / Current Corpus Statistics")
2. The matching table in `RUNBOOK.md` §1.5
3. `WORKLOG.md`, with an entry describing what changed

This keeps the docs in sync with the actual contents of `data/chunks/` instead of letting them drift.
