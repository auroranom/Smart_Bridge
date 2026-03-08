# Smart Bridge — MedSafe AI

A patient-facing healthcare safety web application with 5 AI-assisted modules for medicine safety analysis, prescription parsing, symptom guidance, side-effect monitoring, and emergency risk prediction.

> ⚠️ **Disclaimer**: All AI outputs are **educational only** and are **not** medical diagnoses. Always consult a qualified healthcare professional for medical advice.

## Features

| # | Module | Description |
|---|--------|-------------|
| 1 | **Medicine Interaction Checker** | Check drug-drug interactions with severity ratings and AI safety summaries |
| 2 | **Prescription OCR + AI Parsing** | Upload prescription images for automated text extraction and medicine identification |
| 3 | **Symptom & Doubt Solver** | Get symptom guidance with home remedies, warning signs, and AI explanations |
| 4 | **Side-Effect Monitor** | Analyze medication side effects based on age, gender, and reported symptoms |
| 5 | **Emergency Risk Predictor** | Rule-based risk scoring with transparent breakdowns and safety alerts |

## Tech Stack

- **Python 3.10+**
- **Streamlit** — Multi-tab dashboard UI
- **pytesseract + Pillow** — OCR for prescription images
- **RapidFuzz** — Fuzzy string matching for medicine name resolution
- **Ollama + LLaMA 3** — Local LLM for AI explanations, summarization, and extraction

## Setup Instructions

### 1. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Tesseract OCR

- **Windows**: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and install. Set `MEDSAFE_TESSERACT_CMD` environment variable if installed to a non-default location.
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt install tesseract-ocr`

### 4. Install Ollama and pull a model

```bash
# Install Ollama from https://ollama.com
ollama pull llama3
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at [http://localhost:8501](http://localhost:8501).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEDSAFE_OLLAMA_MODEL` | `llama3` | Fixed Ollama model name |
| `OLLAMA_HOST` | — | Remote Ollama host URL |
| `MEDSAFE_TESSERACT_CMD` | — | Absolute path to `tesseract` binary |
| `MEDSAFE_DISABLE_MODEL_PROBE` | `1` | Set `0` to enable startup model auto-detection |

## Graceful Fallbacks

The app works even without Ollama or Tesseract installed:
- **OCR**: Returns a readable error if Tesseract is missing
- **AI Features**: Falls back to rule-based guidance when LLM is unavailable
- **Medicine Extraction**: Uses local fuzzy matching when LLM is unavailable

## Testing

```bash
# Unit tests
python -m unittest discover -s tests -v

# End-to-end validation
python e2e_validate.py

# Performance benchmarks
python performance_benchmark.py
```

## License

MIT
