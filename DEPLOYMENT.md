# MedSafe AI Deployment Guide

## Supported Targets
- Local server (`streamlit run app.py`)
- Streamlit Cloud (GitHub-connected app deployment)

## Environment Variables
- `MEDSAFE_OLLAMA_MODEL`: optional fixed model name (example: `llama3.2:1b`)
- `OLLAMA_HOST`: optional remote Ollama host URL
- `MEDSAFE_TESSERACT_CMD`: optional absolute path to `tesseract` binary
- `MEDSAFE_DISABLE_MODEL_PROBE`: set `1` (default) to avoid startup model probing delays

## Local Deployment
1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. (Optional) set environment variables listed above.
4. Run:
   - `streamlit run app.py`

## Streamlit Cloud Notes
- Add secrets/environment variables in app settings (or `.streamlit/secrets.toml` locally).
- If Ollama/Tesseract are unavailable, app now falls back safely:
  - OCR paths report readable error state.
  - AI explanation/doubt solver returns educational non-diagnostic fallback.
  - Medicine extraction falls back to local fuzzy extraction when LLM is unavailable.

## Pre-Release Validation Commands
- `python -m unittest discover -s tests -v`
- `python e2e_validate.py`
- `python performance_benchmark.py`
