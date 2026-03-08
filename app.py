from __future__ import annotations

from datetime import datetime
import hashlib
from time import perf_counter

import streamlit as st
from PIL import Image

from med_db import MED_DB, check_interactions, find_medicine, extract_medicines_from_text
from symptom import analyze_symptom, ai_symptom_explanation, analyze_side_effects, ai_doubt_solver
from ocr_utils import (
    dependency_status,
    extract_text_from_image,
    extract_medicines_with_llm,
    validate_medicines_against_db,
)
from risk_engine import calculate_risk_score, apply_safety_rules
from session_logger import log_session_event, get_session_history, get_session_summary, clear_session_log


st.set_page_config(
    page_title="MedSafe AI",
    page_icon="🏥",
    layout="wide"
)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Source+Serif+4:wght@600;700&display=swap');

        :root {
            --ms-bg: #f3f7fb;
            --ms-panel: #ffffff;
            --ms-border: #d6e3ee;
            --ms-text: #14263c;
            --ms-muted: #4f6479;
            --ms-accent: #0f766e;
            --ms-accent-soft: #d6f3ef;
        }

        html, body, [class*="css"] {
            font-family: 'Manrope', 'Segoe UI', sans-serif;
            color: var(--ms-text);
        }

        .stApp {
            background: linear-gradient(145deg, #f3f7fb 0%, #ecf7f7 48%, #f9fbff 100%);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1.2rem;
            padding-bottom: 1.4rem;
        }

        h1, h2, h3 {
            font-family: 'Source Serif 4', Georgia, serif;
        }

        .hero-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid var(--ms-border);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 8px 18px rgba(16, 24, 40, 0.08);
        }

        .hero-card h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.15;
        }

        .hero-card p {
            margin: 0.5rem 0 0;
            color: var(--ms-muted);
            font-size: 0.98rem;
        }

        .panel-title {
            background: var(--ms-panel);
            border: 1px solid var(--ms-border);
            border-radius: 14px;
            padding: 0.75rem 0.95rem;
            margin: 0.2rem 0 1rem;
        }

        .panel-title h3 {
            margin: 0;
            font-size: 1.15rem;
        }

        .panel-title p {
            margin: 0.35rem 0 0;
            color: var(--ms-muted);
            font-size: 0.92rem;
        }

        div[data-testid="stTabs"] button[role="tab"] {
            padding: 0.55rem 0.95rem;
            border-radius: 10px;
            font-weight: 700;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            background: var(--ms-accent-soft);
            color: #0b5d57;
        }

        [data-testid="stHorizontalBlock"] {
            gap: 1rem;
        }

        @media (max-width: 1100px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero-card h1 {
                font-size: 1.65rem;
            }

            .panel-title h3 {
                font-size: 1.02rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def panel_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="panel-title">
            <h3>{title}</h3>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def size_bucket(value: int) -> str:
    if value < 250:
        return "small"
    if value < 1500:
        return "medium"
    return "large"


def combo_bucket(count: int) -> str:
    if count <= 2:
        return "small"
    if count <= 6:
        return "medium"
    return "large"


def log_performance_sample(stage: str, duration_seconds: float, size_label: str, metadata: dict | None = None) -> None:
    samples = st.session_state.get("performance_samples", [])
    samples.append(
        {
            "timestamp": now_str(),
            "stage": stage,
            "duration_ms": round(duration_seconds * 1000, 2),
            "size": size_label,
            "metadata": metadata or {},
        }
    )
    st.session_state["performance_samples"] = samples[-300:]


def summarize_performance(stage: str) -> dict | None:
    samples = [s for s in st.session_state.get("performance_samples", []) if s["stage"] == stage]
    if not samples:
        return None
    values = sorted([s["duration_ms"] for s in samples])
    idx_95 = int(round((len(values) - 1) * 0.95))
    return {
        "count": len(values),
        "avg_ms": round(sum(values) / len(values), 2),
        "p95_ms": values[idx_95],
        "max_ms": max(values),
    }


def uploaded_file_sha256(uploaded_file) -> str:
    payload = uploaded_file.getvalue()
    return hashlib.sha256(payload).hexdigest()


def put_ai_cache(key: str, value: str, max_items: int = 200) -> None:
    cache = st.session_state.get("ai_response_cache", {})
    cache[key] = value
    while len(cache) > max_items:
        first_key = next(iter(cache))
        cache.pop(first_key, None)
    st.session_state["ai_response_cache"] = cache


def init_session_state() -> None:
    if "prescription_data" not in st.session_state:
        st.session_state["prescription_data"] = {
            "filename": None,
            "file_hash": None,
            "processed_at": None,
            "cache_hit": False,
            "timings_ms": {},
            "raw_text": "",
            "extracted": [],
            "validated": [],
            "fuzzy_matches": [],
            "auto_interactions": [],
            "error": None,
        }

    if "interaction_results" not in st.session_state:
        st.session_state["interaction_results"] = {
            "selected_medicines": [],
            "interactions": [],
            "checked_at": None,
            "manual_query": "",
            "manual_match": None,
            "manual_error": None,
        }

    if "side_effect_logs" not in st.session_state:
        st.session_state["side_effect_logs"] = []

    if "risk_analysis_output" not in st.session_state:
        st.session_state["risk_analysis_output"] = None

    if "symptom_result" not in st.session_state:
        st.session_state["symptom_result"] = None

    if "doubt_result" not in st.session_state:
        st.session_state["doubt_result"] = None

    if "ocr_pipeline_cache" not in st.session_state:
        st.session_state["ocr_pipeline_cache"] = {}

    if "ai_response_cache" not in st.session_state:
        st.session_state["ai_response_cache"] = {}

    if "performance_samples" not in st.session_state:
        st.session_state["performance_samples"] = []


def reset_ui_session_data() -> None:
    st.session_state["prescription_data"] = {
        "filename": None,
        "file_hash": None,
        "processed_at": None,
        "cache_hit": False,
        "timings_ms": {},
        "raw_text": "",
        "extracted": [],
        "validated": [],
        "fuzzy_matches": [],
        "auto_interactions": [],
        "error": None,
    }
    st.session_state["interaction_results"] = {
        "selected_medicines": [],
        "interactions": [],
        "checked_at": None,
        "manual_query": "",
        "manual_match": None,
        "manual_error": None,
    }
    st.session_state["side_effect_logs"] = []
    st.session_state["risk_analysis_output"] = None
    st.session_state["symptom_result"] = None
    st.session_state["doubt_result"] = None
    st.session_state["ocr_pipeline_cache"] = {}
    st.session_state["ai_response_cache"] = {}
    st.session_state["performance_samples"] = []


def render_sidebar_session_panel() -> None:
    with st.sidebar:
        st.subheader("🚀 Deployment Status")
        deps = dependency_status()
        if deps.get("has_pytesseract"):
            st.success("OCR backend: pytesseract available")
        else:
            st.warning("OCR backend: pytesseract missing (OCR will fallback)")
        if deps.get("has_ollama"):
            st.success(f"AI backend: Ollama available ({deps.get('ollama_model')})")
        else:
            st.warning("AI backend: Ollama missing (AI responses will fallback)")
        with st.expander("Environment & Dependency Details", expanded=False):
            st.json(deps)

        st.subheader("📋 Session Snapshot")
        summary = get_session_summary()
        st.metric("Events", summary.get("total_events", 0))

        first_event = summary.get("first_event")
        last_event = summary.get("last_event")
        if first_event and last_event:
            st.caption(f"{first_event} → {last_event}")

        with st.expander("Recent activity", expanded=False):
            history = get_session_history(limit=8)
            if history:
                for entry in reversed(history):
                    event_name = entry["event"].replace("_", " ").title()
                    st.write(f"**{entry['timestamp']}**")
                    st.write(event_name)
                    st.caption(str(entry.get("data", {})))
                    st.divider()
            else:
                st.caption("No activity logged yet.")

        if st.button("🗑️ Clear Session Log", key="clear_session_log_sidebar"):
            clear_session_log()
            st.success("Session log cleared")
            st.rerun()

        if st.button("♻️ Reset UI Session Data", key="reset_ui_session_data"):
            reset_ui_session_data()
            st.success("In-memory UI data reset")
            st.rerun()

        with st.expander("⚡ Performance Summary", expanded=False):
            tracked_stages = [
                "ocr_total",
                "ocr_extract_text",
                "ocr_extract_medicines_llm",
                "fuzzy_text_scan",
                "interaction_check",
                "symptom_ai",
                "doubt_ai",
                "risk_score",
            ]
            found_any = False
            for stage in tracked_stages:
                stats = summarize_performance(stage)
                if not stats:
                    continue
                found_any = True
                st.caption(
                    f"{stage}: n={stats['count']}, avg={stats['avg_ms']} ms, "
                    f"p95={stats['p95_ms']} ms, max={stats['max_ms']} ms"
                )
            if not found_any:
                st.caption("No performance samples recorded yet.")

            if st.button("Clear Performance Samples", key="clear_perf_samples"):
                st.session_state["performance_samples"] = []
                st.success("Performance samples cleared")
                st.rerun()


def render_interaction_details(matched: str) -> None:
    db_info = MED_DB[matched]
    st.success(f"Found in database: {matched.capitalize()}")
    st.write(f"Description: {db_info['description']}")
    st.write(f"Category: {db_info.get('category', 'N/A')}")
    st.write(f"Side effects: {', '.join(db_info.get('side_effects', []))}")
    known_interactions = db_info.get("interactions", {})
    if known_interactions:
        st.write("Known interactions:")
        for drug, severity in known_interactions.items():
            st.warning(f"↔ {drug.capitalize()}: {severity}")


def render_risk_severity_alert(category: str) -> None:
    if category == "Low":
        st.success("Severity Level: Low")
    elif category == "Medium":
        st.warning("Severity Level: Medium")
    elif category == "High":
        st.error("Severity Level: High")
    else:
        st.error("Severity Level: Critical")


init_session_state()
inject_custom_css()
render_sidebar_session_panel()

st.markdown(
    """
    <div class="hero-card">
        <h1>MedSafe AI Clinical Dashboard</h1>
        <p>Structured decision-support workspace for medicine safety checks, OCR-based prescription parsing, symptom guidance, side-effect monitoring, and emergency risk prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Educational assistant only. This tool does not replace licensed medical advice.")

medicine_options = [m.capitalize() for m in MED_DB.keys()]

(
    tab_ocr,
    tab_interactions,
    tab_symptoms,
    tab_side_effects,
    tab_emergency,
) = st.tabs(
    [
        "🧾 Prescription OCR",
        "💊 Medicine Interaction Checker",
        "🤒 Symptom & Doubt Solver",
        "⚠️ Side-Effect Monitor",
        "🚨 Emergency Risk Predictor",
    ]
)


with tab_ocr:
    panel_header(
        "Prescription OCR",
        "Upload a prescription image, extract medicines via OCR + AI parsing, and auto-check interaction risks.",
    )

    uploaded_file = st.file_uploader(
        "Choose prescription image",
        type=["jpg", "jpeg", "png"],
        key="prescription_upload",
    )

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            preview_image = Image.open(uploaded_file)
            st.image(preview_image, caption=f"Selected file: {uploaded_file.name}", use_container_width=True)
        except Exception as exc:
            st.error(f"Could not preview image file: {exc}")

    if st.button("🧠 Process Prescription", key="process_prescription_btn"):
        if uploaded_file is None:
            st.warning("Please upload a prescription image before processing.")
        else:
            try:
                total_start = perf_counter()
                file_hash = uploaded_file_sha256(uploaded_file)
                cached_payload = st.session_state["ocr_pipeline_cache"].get(file_hash)

                if cached_payload:
                    cached_result = dict(cached_payload)
                    cached_result["filename"] = uploaded_file.name
                    cached_result["file_hash"] = file_hash
                    cached_result["processed_at"] = now_str()
                    cached_result["cache_hit"] = True
                    st.session_state["prescription_data"] = cached_result

                    raw_len = len(cached_result.get("raw_text", ""))
                    log_performance_sample(
                        "ocr_total",
                        perf_counter() - total_start,
                        size_bucket(raw_len),
                        {"cache_hit": True, "raw_text_length": raw_len},
                    )
                    st.success("Loaded cached OCR pipeline output for this prescription image.")
                else:
                    uploaded_file.seek(0)
                    image = Image.open(uploaded_file)

                    with st.spinner("Running OCR and medicine parsing..."):
                        ocr_start = perf_counter()
                        raw_text = extract_text_from_image(image)
                        ocr_duration = perf_counter() - ocr_start

                        if raw_text.startswith("Error extracting text:"):
                            raise RuntimeError(raw_text)

                        if not raw_text.strip():
                            raise ValueError("OCR returned empty text. Upload a clearer image.")

                        llm_start = perf_counter()
                        extracted = extract_medicines_with_llm(raw_text)
                        llm_duration = perf_counter() - llm_start
                        if not isinstance(extracted, list):
                            extracted = []

                        validated = validate_medicines_against_db(extracted, MED_DB)

                        fuzzy_start = perf_counter()
                        fuzzy_matches = extract_medicines_from_text(raw_text)
                        fuzzy_duration = perf_counter() - fuzzy_start

                        interaction_start = perf_counter()
                        med_keys = [item["matched_db_key"] for item in validated if item.get("matched_db_key")]
                        auto_interactions = check_interactions(med_keys) if len(med_keys) > 1 else []
                        interaction_duration = perf_counter() - interaction_start

                    total_duration = perf_counter() - total_start
                    raw_len = len(raw_text)
                    timings_ms = {
                        "ocr_extract_text_ms": round(ocr_duration * 1000, 2),
                        "llm_extract_medicines_ms": round(llm_duration * 1000, 2),
                        "fuzzy_scan_ms": round(fuzzy_duration * 1000, 2),
                        "interaction_check_ms": round(interaction_duration * 1000, 2),
                        "total_ms": round(total_duration * 1000, 2),
                    }

                    fresh_result = {
                        "filename": uploaded_file.name,
                        "file_hash": file_hash,
                        "processed_at": now_str(),
                        "cache_hit": False,
                        "timings_ms": timings_ms,
                        "raw_text": raw_text,
                        "extracted": extracted,
                        "validated": validated,
                        "fuzzy_matches": fuzzy_matches,
                        "auto_interactions": auto_interactions,
                        "error": None,
                    }
                    st.session_state["prescription_data"] = fresh_result
                    st.session_state["ocr_pipeline_cache"][file_hash] = dict(fresh_result)

                    size = size_bucket(raw_len)
                    log_performance_sample("ocr_extract_text", ocr_duration, size, {"raw_text_length": raw_len})
                    log_performance_sample("ocr_extract_medicines_llm", llm_duration, size, {"raw_text_length": raw_len})
                    log_performance_sample("fuzzy_text_scan", fuzzy_duration, size, {"raw_text_length": raw_len})
                    log_performance_sample("interaction_check", interaction_duration, size, {"source": "ocr_auto"})
                    log_performance_sample("ocr_total", total_duration, size, {"cache_hit": False, "raw_text_length": raw_len})

                log_session_event(
                    "prescription_upload",
                    {
                        "filename": uploaded_file.name,
                        "medicines_detected": [
                            item.get("medicine")
                            for item in st.session_state["prescription_data"].get("validated", [])
                        ],
                        "raw_text_length": len(st.session_state["prescription_data"].get("raw_text", "")),
                    },
                )
                st.success("Prescription processed successfully.")

            except Exception as exc:
                st.session_state["prescription_data"]["error"] = str(exc)
                st.error(f"Failed to process prescription: {exc}")

    prescription_state = st.session_state["prescription_data"]
    if prescription_state.get("processed_at"):
        cache_hit_label = "Yes" if prescription_state.get("cache_hit") else "No"
        st.caption(
            f"Last processed: {prescription_state['processed_at']}"
            f" | File: {prescription_state.get('filename') or 'N/A'}"
            f" | Cache Hit: {cache_hit_label}"
        )
    timings_ms = prescription_state.get("timings_ms", {})
    if timings_ms:
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("OCR (ms)", timings_ms.get("ocr_extract_text_ms", 0))
        t2.metric("LLM (ms)", timings_ms.get("llm_extract_medicines_ms", 0))
        t3.metric("Fuzzy (ms)", timings_ms.get("fuzzy_scan_ms", 0))
        t4.metric("Interaction (ms)", timings_ms.get("interaction_check_ms", 0))
        t5.metric("Total (ms)", timings_ms.get("total_ms", 0))

    if prescription_state.get("error"):
        st.error(f"Last processing error: {prescription_state['error']}")

    if prescription_state.get("raw_text"):
        validated = prescription_state.get("validated", [])
        extracted = prescription_state.get("extracted", [])
        fuzzy_text_meds = prescription_state.get("fuzzy_matches", [])
        auto_interactions = prescription_state.get("auto_interactions", [])
        in_db_count = sum(1 for item in validated if item.get("in_database"))

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Medicines Extracted", len(extracted))
        metric_col2.metric("Matched in Database", in_db_count)
        metric_col3.metric("Interaction Flags", len(auto_interactions))

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Raw OCR Text")
            st.text_area(
                "Extracted text",
                prescription_state["raw_text"],
                height=220,
                disabled=True,
            )
            with st.expander("Show OCR Text (Monospace)"):
                st.code(prescription_state["raw_text"] or "", language="text")

        with col_right:
            st.markdown("#### Extracted Medicines")
            if validated:
                for item in validated:
                    med_name = item.get("medicine", "Unknown")
                    salt = item.get("salt") or "Unknown salt"
                    in_db = item.get("in_database", False)
                    matched = item.get("matched_db_key")

                    if in_db:
                        st.success(f"{med_name} | {salt}")
                    else:
                        st.warning(f"{med_name} | {salt} (not in local DB)")

                    if in_db and matched:
                        db_entry = MED_DB[matched]
                        with st.expander(f"Details: {matched.capitalize()}"):
                            st.write(f"Category: {db_entry.get('category', 'N/A')}")
                            st.write(f"Description: {db_entry.get('description', 'N/A')}")
                            dose = db_entry.get("standard_dose_mg", {})
                            st.write(f"Typical adult dose: {dose.get('adult', 'N/A')} mg")
                            st.write(f"Known side effects: {', '.join(db_entry.get('side_effects', []))}")
            else:
                st.info("No medicines parsed from the last OCR run.")

        st.markdown("#### Auto Interaction Check")
        if auto_interactions:
            for interaction in auto_interactions:
                st.error(interaction)
        elif len(validated) > 1:
            st.success("No interaction warnings from parsed medicines.")
        else:
            st.info("At least two medicines are needed for interaction checks.")

        st.markdown("#### Fuzzy Scan from OCR Text")
        if fuzzy_text_meds:
            st.success(f"Detected from raw text: {', '.join([m.capitalize() for m in fuzzy_text_meds])}")
        else:
            st.warning("No medicine detected from fuzzy text scan.")

        with st.expander("Structured Extraction JSON", expanded=False):
            st.json({
                "filename": prescription_state.get("filename"),
                "file_hash": prescription_state.get("file_hash"),
                "processed_at": prescription_state.get("processed_at"),
                "cache_hit": prescription_state.get("cache_hit", False),
                "timings_ms": prescription_state.get("timings_ms", {}),
                "llm_extracted": extracted,
                "validated_medicines": validated,
                "fuzzy_matches": fuzzy_text_meds,
                "auto_interactions": auto_interactions,
            })


with tab_interactions:
    panel_header(
        "Medicine Interaction Checker",
        "Select medicines or use fuzzy manual search to identify known interaction risks and key medicine details.",
    )

    selected_meds = st.multiselect(
        "Select medicines",
        medicine_options,
        help="Choose at least two medicines for cross-checking",
        key="interaction_multiselect",
    )

    manual_med = st.text_input(
        "Manual medicine lookup (fuzzy search)",
        key="manual_medicine_lookup",
    )

    lookup_col, check_col = st.columns(2)

    with lookup_col:
        if st.button("📚 Lookup Medicine", key="lookup_medicine_btn"):
            query = (manual_med or "").strip()
            if not query:
                st.warning("Enter a medicine name before lookup.")
                st.session_state["interaction_results"]["manual_error"] = "Empty medicine query"
                st.session_state["interaction_results"]["manual_match"] = None
            else:
                matched = find_medicine(query)
                st.session_state["interaction_results"]["manual_query"] = query
                st.session_state["interaction_results"]["manual_match"] = matched
                st.session_state["interaction_results"]["manual_error"] = None if matched else "Medicine not found"

    with check_col:
        if st.button("🔍 Check Selected Interactions", key="check_interactions_button"):
            if len(selected_meds) < 2:
                st.warning("Select at least two medicines to run interaction checks.")
            else:
                try:
                    start = perf_counter()
                    meds_lower = [m.lower() for m in selected_meds]
                    interactions = check_interactions(meds_lower)
                    duration = perf_counter() - start

                    st.session_state["interaction_results"].update(
                        {
                            "selected_medicines": selected_meds,
                            "interactions": interactions,
                            "checked_at": now_str(),
                        }
                    )

                    log_session_event(
                        "interaction_check",
                        {
                            "medicines": selected_meds,
                            "interactions_found": len(interactions),
                        },
                    )
                    log_performance_sample(
                        "interaction_check",
                        duration,
                        combo_bucket(len(selected_meds)),
                        {"source": "interaction_tab", "med_count": len(selected_meds)},
                    )
                except Exception as exc:
                    st.error(f"Could not complete interaction check: {exc}")

    interaction_state = st.session_state["interaction_results"]

    if interaction_state.get("manual_error"):
        st.warning(f"Lookup status: {interaction_state['manual_error']}")

    if interaction_state.get("manual_match"):
        render_interaction_details(interaction_state["manual_match"])

    if interaction_state.get("checked_at"):
        st.caption(f"Last interaction check: {interaction_state['checked_at']}")

    persisted_interactions = interaction_state.get("interactions", [])
    persisted_meds = interaction_state.get("selected_medicines", [])
    if persisted_meds:
        m1, m2, m3 = st.columns(3)
        m1.metric("Medicines Checked", len(persisted_meds))
        m2.metric("Warnings Found", len(persisted_interactions))
        m3.metric("Status", "Alert" if persisted_interactions else "Clear")

        st.write(f"Medicines checked: {', '.join(persisted_meds)}")
        if persisted_interactions:
            st.error("Potential interactions found")
            for interaction in persisted_interactions:
                st.markdown(f"- {interaction}")
        else:
            st.success("No known interactions found for selected medicines.")

        with st.expander("Interaction Result JSON", expanded=False):
            st.json({
                "checked_at": interaction_state.get("checked_at"),
                "selected_medicines": persisted_meds,
                "interactions": persisted_interactions,
                "manual_lookup_query": interaction_state.get("manual_query"),
                "manual_lookup_match": interaction_state.get("manual_match"),
            })


with tab_symptoms:
    panel_header(
        "Symptom & Doubt Solver",
        "Get quick rule-based symptom guidance, AI educational explanations, and non-diagnostic answers to common health doubts.",
    )

    col_symptom, col_doubt = st.columns(2)

    with col_symptom:
        st.markdown("#### Symptom Analyzer")
        symptom_input = st.text_input(
            "Enter symptom",
            placeholder="e.g., headache, fever, rash",
            key="symptom_input",
        )

        if st.button("Analyze Symptom", key="analyze_symptom_button"):
            symptom_text = (symptom_input or "").strip()
            if not symptom_text:
                st.warning("Please enter a symptom before analysis.")
            else:
                try:
                    rule_advice = analyze_symptom(symptom_text)
                    cache_key = f"symptom::{symptom_text.lower()}"
                    ai_cache = st.session_state.get("ai_response_cache", {})
                    ai_cache_hit = cache_key in ai_cache
                    ai_duration = 0.0

                    if ai_cache_hit:
                        ai_advice = ai_cache[cache_key]
                    else:
                        with st.spinner("Generating explanation..."):
                            ai_start = perf_counter()
                            ai_advice = ai_symptom_explanation(symptom_text)
                            ai_duration = perf_counter() - ai_start
                        put_ai_cache(cache_key, ai_advice)

                    log_performance_sample(
                        "symptom_ai",
                        ai_duration,
                        size_bucket(len(symptom_text)),
                        {"cache_hit": ai_cache_hit, "input_length": len(symptom_text)},
                    )

                    st.session_state["symptom_result"] = {
                        "symptom": symptom_text,
                        "rule_advice": rule_advice,
                        "ai_advice": ai_advice,
                        "ai_cache_hit": ai_cache_hit,
                        "ai_latency_ms": round(ai_duration * 1000, 2),
                        "generated_at": now_str(),
                    }

                    log_session_event("symptom_analysis", {"symptom": symptom_text})
                except Exception as exc:
                    st.error(f"Could not analyze symptom: {exc}")

        symptom_state = st.session_state.get("symptom_result")
        if symptom_state:
            st.caption(f"Last analyzed: {symptom_state['generated_at']}")
            st.caption(
                f"AI cache hit: {'Yes' if symptom_state.get('ai_cache_hit') else 'No'}"
                f" | AI latency: {symptom_state.get('ai_latency_ms', 0)} ms"
            )
            st.markdown("#### Symptom Guidance")
            if "not recognized" in symptom_state["rule_advice"].lower():
                st.warning(symptom_state["rule_advice"])
            else:
                st.info(symptom_state["rule_advice"])

            with st.expander("AI Educational Explanation", expanded=True):
                st.success(symptom_state["ai_advice"])

            with st.expander("Symptom Analysis JSON", expanded=False):
                st.json(symptom_state)
            st.caption("Educational output only. Not a medical diagnosis.")

    with col_doubt:
        st.markdown("#### Health Doubt Solver")
        doubt_input = st.text_area(
            "Ask a health-related question",
            placeholder="e.g., Is mild fever after vaccination common?",
            height=130,
            key="doubt_input",
        )

        if st.button("Resolve Doubt", key="resolve_doubt_button"):
            question = (doubt_input or "").strip()
            if not question:
                st.warning("Please enter a health question before submitting.")
            else:
                try:
                    cache_key = f"doubt::{question.lower()}"
                    ai_cache = st.session_state.get("ai_response_cache", {})
                    ai_cache_hit = cache_key in ai_cache
                    ai_duration = 0.0

                    if ai_cache_hit:
                        doubt_answer = ai_cache[cache_key]
                    else:
                        with st.spinner("Preparing educational response..."):
                            ai_start = perf_counter()
                            doubt_answer = ai_doubt_solver(question)
                            ai_duration = perf_counter() - ai_start
                        put_ai_cache(cache_key, doubt_answer)

                    log_performance_sample(
                        "doubt_ai",
                        ai_duration,
                        size_bucket(len(question)),
                        {"cache_hit": ai_cache_hit, "input_length": len(question)},
                    )

                    st.session_state["doubt_result"] = {
                        "question": question,
                        "answer": doubt_answer,
                        "ai_cache_hit": ai_cache_hit,
                        "ai_latency_ms": round(ai_duration * 1000, 2),
                        "generated_at": now_str(),
                    }

                    log_session_event("doubt_solver", {"question": question})
                except Exception as exc:
                    st.error(f"Could not resolve doubt: {exc}")

        doubt_state = st.session_state.get("doubt_result")
        if doubt_state:
            st.caption(f"Last answered: {doubt_state['generated_at']}")
            st.caption(
                f"AI cache hit: {'Yes' if doubt_state.get('ai_cache_hit') else 'No'}"
                f" | AI latency: {doubt_state.get('ai_latency_ms', 0)} ms"
            )
            st.markdown("#### Doubt Solver Output")
            st.info(f"Question: {doubt_state['question']}")
            with st.expander("AI Answer", expanded=True):
                st.success(doubt_state["answer"])
            with st.expander("Doubt Solver JSON", expanded=False):
                st.json(doubt_state)
            st.caption("This response is educational and non-diagnostic.")


with tab_side_effects:
    panel_header(
        "Side-Effect Monitor",
        "Assess medicine side-effect risk against patient age, gender, and reported symptoms.",
    )

    se_meds = st.multiselect(
        "Current medicines",
        medicine_options,
        key="side_effect_medicines",
    )
    se_age = st.number_input(
        "Patient age",
        min_value=0,
        max_value=120,
        value=30,
        key="side_effect_age",
    )
    se_gender = st.selectbox(
        "Patient gender",
        ["Male", "Female", "Other"],
        key="side_effect_gender",
    )
    se_symptom = st.text_input(
        "Reported symptom/experience",
        key="side_effect_reported_symptom",
    )

    if st.button("Check Side-Effect Risk", key="check_side_effect_risk"):
        symptom_text = (se_symptom or "").strip()
        if not se_meds:
            st.warning("Select at least one medicine to run side-effect monitoring.")
        elif not symptom_text:
            st.warning("Enter the reported symptom/experience to evaluate side-effect overlap.")
        else:
            try:
                meds_lower = [m.lower() for m in se_meds]
                warnings = analyze_side_effects(meds_lower, int(se_age), se_gender, symptom_text)
                entry = {
                    "timestamp": now_str(),
                    "medicines": se_meds,
                    "age": int(se_age),
                    "gender": se_gender,
                    "symptom": symptom_text,
                    "warnings": warnings,
                }

                st.session_state["side_effect_logs"].append(entry)
                st.session_state["side_effect_logs"] = st.session_state["side_effect_logs"][-20:]

                log_session_event(
                    "side_effect_check",
                    {
                        "medicines": se_meds,
                        "age": int(se_age),
                        "gender": se_gender,
                        "symptom": symptom_text,
                        "warnings_count": len(warnings),
                    },
                )
            except Exception as exc:
                st.error(f"Could not evaluate side-effect risk: {exc}")

    side_effect_logs = st.session_state.get("side_effect_logs", [])
    if side_effect_logs:
        latest = side_effect_logs[-1]
        st.caption(f"Last check: {latest['timestamp']}")
        sev_errors = sum(1 for w in latest["warnings"] if "🔴" in w or "🚨" in w)
        sev_warnings = sum(1 for w in latest["warnings"] if "⚠️" in w)
        sev_ok = len(latest["warnings"]) - sev_errors - sev_warnings
        c1, c2, c3 = st.columns(3)
        c1.metric("Critical Alerts", sev_errors)
        c2.metric("Warnings", sev_warnings)
        c3.metric("Informational", max(sev_ok, 0))

        for warning in latest["warnings"]:
            if "🔴" in warning or "🚨" in warning:
                st.error(warning)
            elif "⚠️" in warning:
                st.warning(warning)
            else:
                st.success(warning)

        with st.expander("Latest Side-Effect Check JSON", expanded=False):
            st.json(latest)

        with st.expander("Previous side-effect checks", expanded=False):
            for entry in reversed(side_effect_logs[:-1]):
                st.write(
                    f"**{entry['timestamp']}** | Medicines: {', '.join(entry['medicines'])}"
                    f" | Symptom: {entry['symptom']}"
                )
                for warning in entry["warnings"]:
                    st.caption(warning)
                st.divider()


with tab_emergency:
    panel_header(
        "Emergency Risk Predictor",
        "Compute a transparent rule-based emergency risk percentage with factor-by-factor scoring and safety alerts.",
    )

    col_input, col_output = st.columns(2)

    with col_input:
        rs_age = st.number_input(
            "Patient age",
            min_value=0,
            max_value=120,
            value=30,
            key="risk_age",
        )
        rs_severity = st.selectbox(
            "Symptom severity",
            ["Low", "Medium", "High", "Critical"],
            key="risk_severity",
        )
        rs_meds = st.multiselect(
            "Current medicines",
            medicine_options,
            key="risk_medicines",
        )
        rs_conditions = st.multiselect(
            "Chronic conditions",
            [
                "Diabetes",
                "Heart Disease",
                "Kidney Disease",
                "Liver Disease",
                "Hypertension",
                "Cancer",
                "None",
            ],
            key="risk_conditions",
        )

    with col_output:
        st.caption("Run the predictor to view risk percentage, category, and rationale.")
        if st.button("⚡ Calculate Emergency Risk", key="calculate_risk_button"):
            if "None" in rs_conditions and len(rs_conditions) > 1:
                st.warning("Select either 'None' or specific chronic conditions, not both.")
            else:
                try:
                    risk_start = perf_counter()
                    meds_lower = [m.lower() for m in rs_meds]
                    conds = [c for c in rs_conditions if c != "None"]

                    result = calculate_risk_score(
                        age=int(rs_age),
                        severity=rs_severity,
                        medicines=meds_lower,
                        chronic_conditions=conds,
                    )
                    risk_duration = perf_counter() - risk_start
                    safety_alerts = apply_safety_rules(int(rs_age), meds_lower, rs_severity)
                    log_performance_sample(
                        "risk_score",
                        risk_duration,
                        combo_bucket(len(rs_meds)),
                        {
                            "med_count": len(rs_meds),
                            "condition_count": len(conds),
                            "severity": rs_severity,
                        },
                    )

                    st.session_state["risk_analysis_output"] = {
                        "timestamp": now_str(),
                        "inputs": {
                            "age": int(rs_age),
                            "severity": rs_severity,
                            "medicines": rs_meds,
                            "conditions": conds,
                        },
                        "result": result,
                        "safety_alerts": safety_alerts,
                    }

                    log_session_event(
                        "risk_score",
                        {
                            "age": int(rs_age),
                            "severity": rs_severity,
                            "medicines": rs_meds,
                            "conditions": conds,
                            "percentage": result["percentage"],
                            "category": result["category"],
                        },
                    )
                except Exception as exc:
                    st.error(f"Could not calculate risk score: {exc}")

        risk_state = st.session_state.get("risk_analysis_output")
        if risk_state:
            result = risk_state["result"]
            percentage = result["percentage"]
            category = result["category"]
            severity_rank = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

            st.caption(f"Last risk analysis: {risk_state['timestamp']}")
            st.write(
                "Inputs: "
                f"Age {risk_state['inputs']['age']}, "
                f"Severity {risk_state['inputs']['severity']}, "
                f"Medicines {', '.join(risk_state['inputs']['medicines']) or 'None'}, "
                f"Conditions {', '.join(risk_state['inputs']['conditions']) or 'None'}"
            )

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Risk Percentage", f"{percentage}%")
            col_b.metric("Severity Level", category)
            col_c.metric("Severity Rank (1-4)", severity_rank.get(category, 0))
            st.progress(float(percentage) / 100)
            render_risk_severity_alert(category)

            if category == "Low":
                st.success("Risk level: Low. Routine monitoring recommended.")
            elif category == "Medium":
                st.warning("Risk level: Medium. Monitor closely and consult a doctor.")
            elif category == "High":
                st.error("Risk level: High. Prompt medical attention recommended.")
            else:
                st.error("Risk level: Critical. Seek emergency care immediately.")

            st.markdown("#### Score Breakdown")
            for factor, pts in result["breakdown"].items():
                st.write(f"**{factor}:** {pts}")
            st.write(f"**Total:** {result['score']} / {result['max_score']} points")

            safety_alerts = risk_state.get("safety_alerts", [])
            if safety_alerts:
                st.markdown("#### Safety Rule Alerts")
                for alert in safety_alerts:
                    st.error(alert)
            else:
                st.success("No additional safety-rule alerts triggered.")

            with st.expander("Risk Analysis JSON", expanded=False):
                st.json(risk_state)
