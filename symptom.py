try:
    from rapidfuzz import process, fuzz
except ImportError:
    from difflib import SequenceMatcher

    class _FallbackFuzz:
        @staticmethod
        def WRatio(a, b):
            return int(SequenceMatcher(None, a, b).ratio() * 100)

    class _FallbackProcess:
        @staticmethod
        def extractOne(query, choices, scorer=None):
            best_choice = None
            best_score = -1
            scorer = scorer or _FallbackFuzz.WRatio
            for choice in choices:
                score = scorer(query, choice)
                if score > best_score:
                    best_choice = choice
                    best_score = score
            if best_choice is None:
                return None
            return (best_choice, best_score, None)

    process = _FallbackProcess()
    fuzz = _FallbackFuzz()

try:
    import ollama  # type: ignore
except Exception:  # pragma: no cover - import guard
    ollama = None

from ocr_utils import OLLAMA_MODEL, HAS_OLLAMA

EDUCATIONAL_DISCLAIMER = (
    "Educational information only. This is not a diagnosis or medical advice."
)

# ─────────────────────────────────────────────
#  RULE-BASED SYMPTOM ADVICE DATABASE
# ─────────────────────────────────────────────
symptom_advice = {
    "headache": "Rest in a quiet, dark room. Drink plenty of water. Try a cold compress on forehead. Consult a doctor if severe or sudden.",
    "fever": "Stay hydrated. Rest and use a light blanket. Take fever-reducing medication if needed. See a doctor if it exceeds 103°F (39.4°C).",
    "cough": "Drink warm fluids. Use honey if over 1 year old. Rest. See a doctor if it lasts more than 3 weeks or produces blood.",
    "nausea": "Eat small bland meals. Avoid strong smells. Stay hydrated. Seek help if you cannot keep liquids down for 24 hours.",
    "chest pain": "URGENT: If accompanied by shortness of breath, arm/jaw pain, or sweating — call emergency services immediately.",
    "dizziness": "Sit or lie down immediately. Avoid sudden movements. Drink water. See a doctor if it persists or is accompanied by hearing loss.",
    "fatigue": "Rest and sleep adequately. Maintain a balanced diet. Avoid caffeine overuse. See a doctor if fatigue is persistent and unexplained.",
    "diarrhea": "Stay well hydrated with ORS (Oral Rehydration Solution). Avoid dairy and spicy foods. See a doctor if it lasts over 2 days.",
    "vomiting": "Sip clear fluids slowly. Avoid solid food for a few hours. See a doctor if it persists over 24 hours or blood is present.",
    "rash": "Keep area clean and dry. Avoid scratching. Use calamine lotion. See a doctor if spreading rapidly or accompanied by fever.",
    "shortness of breath": "URGENT: Sit upright and stay calm. If sudden or severe, call emergency services immediately.",
    "back pain": "Apply heat or cold pack. Gentle stretching can help. Avoid heavy lifting. Consult a doctor if it radiates down the leg.",
    "sore throat": "Gargle with warm salt water. Drink warm fluids. Use throat lozenges. See a doctor if it lasts more than a week.",
    "stomach pain": "Avoid spicy/fatty foods. Apply warm compress. Stay hydrated. See a doctor if severe or accompanied by fever.",
    "joint pain": "Rest the affected joint. Apply ice pack. Over-the-counter NSAIDs may help. See a doctor if swelling or fever present."
}

# ─────────────────────────────────────────────
#  SIDE EFFECT RISK DATABASE
# ─────────────────────────────────────────────
SIDE_EFFECT_RISK = {
    "warfarin": {
        "high_risk_age": [">65", "<12"],
        "high_risk_gender": None,
        "warnings": "Warfarin has very narrow therapeutic range. Older patients have significantly increased bleeding risk. Regular INR monitoring essential.",
        "dangerous_combinations": ["aspirin", "ibuprofen", "paracetamol"]
    },
    "metformin": {
        "high_risk_age": [">80"],
        "high_risk_gender": None,
        "warnings": "Risk of lactic acidosis increases in elderly. Avoid alcohol. Pause before contrast imaging procedures.",
        "dangerous_combinations": ["alcohol"]
    },
    "ibuprofen": {
        "high_risk_age": [">65", "<6"],
        "high_risk_gender": None,
        "warnings": "NSAIDs increase risk of GI bleeding especially in elderly. Avoid in kidney disease patients.",
        "dangerous_combinations": ["aspirin", "warfarin"]
    },
    "aspirin": {
        "high_risk_age": ["<16"],
        "high_risk_gender": None,
        "warnings": "Do NOT give aspirin to children under 16 due to risk of Reye's syndrome.",
        "dangerous_combinations": ["warfarin", "ibuprofen"]
    },
    "atorvastatin": {
        "high_risk_age": None,
        "high_risk_gender": "female",
        "warnings": "Women of childbearing age must avoid — teratogenic risk. Report muscle pain immediately as it can indicate myopathy.",
        "dangerous_combinations": ["clarithromycin"]
    }
}


# ─────────────────────────────────────────────
#  RULE-BASED SYMPTOM ANALYZER
# ─────────────────────────────────────────────
def analyze_symptom(symptom: str) -> str:
    """
    Analyzes a symptom using fuzzy matching and returns rule-based advice.
    """
    symptom = symptom.lower().strip()

    match = process.extractOne(symptom, symptom_advice.keys(), scorer=fuzz.WRatio)

    if match and match[1] >= 80:
        matched_symptom = match[0]
        return f"({matched_symptom.capitalize()}) " + symptom_advice[matched_symptom]

    return "Symptom not recognized. Please consult a healthcare professional for advice."


# ─────────────────────────────────────────────
#  AI-ENHANCED SYMPTOM EXPLANATION (LLM)
# ─────────────────────────────────────────────
def ai_symptom_explanation(symptom: str) -> str:
    """
    Uses LLaMA 3 to generate an educational explanation for a symptom,
    including home remedies, lifestyle suggestions, and warning signs.
    Non-diagnostic, educational output only.
    """
    prompt = f"""You are a friendly, non-diagnostic medical education assistant.

A user reports the symptom: "{symptom}"

Provide a brief, helpful, and educational response covering:
1. What this symptom commonly indicates (general causes)
2. 2-3 simple home remedies or comfort measures
3. Lifestyle suggestions that may help
4. Warning signs that mean they should see a doctor immediately

Keep it concise (under 150 words), friendly, and always remind the user this is educational only and not medical advice.
Do NOT diagnose. Do NOT prescribe."""

    if not HAS_OLLAMA or ollama is None:
        return (
            "AI explanation service is currently unavailable. "
            "Use the rule-based guidance shown above and consult a clinician if symptoms worsen.\n\n"
            f"{EDUCATIONAL_DISCLAIMER}"
        )

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response["message"]["content"].strip()
        if "not a diagnosis" not in content.lower() and "not medical advice" not in content.lower():
            content = f"{content}\n\n{EDUCATIONAL_DISCLAIMER}"
        return content
    except Exception as e:
        return f"AI explanation unavailable: {e}\n\n{EDUCATIONAL_DISCLAIMER}"


def ai_doubt_solver(question: str) -> str:
    """
    Uses LLM to answer general health doubts in an educational,
    non-diagnostic way.
    """
    prompt = f"""You are an educational health assistant.

User question:
"{question}"

Respond with:
1. A clear plain-language explanation
2. Practical, low-risk self-care guidance (if relevant)
3. Red-flag symptoms that need urgent medical care

Rules:
- Educational only, not a diagnosis
- Do not prescribe medication doses
- Keep response under 170 words
- End with a short reminder to consult a qualified clinician for personal care decisions
"""
    if not HAS_OLLAMA or ollama is None:
        return (
            "AI doubt solver is currently unavailable. "
            "Please refer to trusted medical sources and seek personalized clinical advice.\n\n"
            f"{EDUCATIONAL_DISCLAIMER}"
        )

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        content = response["message"]["content"].strip()
        if "not a diagnosis" not in content.lower() and "not medical advice" not in content.lower():
            content = f"{content}\n\n{EDUCATIONAL_DISCLAIMER}"
        return content
    except Exception as e:
        return f"Doubt solver unavailable: {e}\n\n{EDUCATIONAL_DISCLAIMER}"


# ─────────────────────────────────────────────
#  SIDE-EFFECT MONITOR
# ─────────────────────────────────────────────
def analyze_side_effects(medicines: list, age: int, gender: str, reported_symptom: str) -> list:
    """
    Analyzes potential side effects for given medicines based on
    patient age, gender, and reported symptoms.
    Returns a list of warning strings.
    """
    warnings = []

    for med in medicines:
        med = med.lower().strip()
        if med not in SIDE_EFFECT_RISK:
            continue

        risk = SIDE_EFFECT_RISK[med]

        # Age-based risk check
        if risk["high_risk_age"]:
            for age_range in risk["high_risk_age"]:
                if age_range.startswith(">") and age > int(age_range[1:]):
                    warnings.append(f"⚠️ {med.capitalize()}: HIGH RISK for age {age} — {risk['warnings']}")
                    break
                elif age_range.startswith("<") and age < int(age_range[1:]):
                    warnings.append(f"⚠️ {med.capitalize()}: HIGH RISK for age {age} — {risk['warnings']}")
                    break

        # Gender-based risk check
        if risk["high_risk_gender"] and gender.lower() == risk["high_risk_gender"]:
            warnings.append(f"⚠️ {med.capitalize()}: Gender-specific risk ({gender}) — {risk['warnings']}")

        # Symptom matches known side effect
        from med_db import MED_DB
        for med_key, med_data in MED_DB.items():
            if med_key == med:
                for side_effect in med_data.get("side_effects", []):
                    if side_effect.lower() in reported_symptom.lower():
                        warnings.append(
                            f"🔴 {med.capitalize()}: Reported symptom '{reported_symptom}' matches known side effect '{side_effect}'. Consult your doctor."
                        )

    return warnings if warnings else ["✅ No specific side-effect risks detected for this profile."]
