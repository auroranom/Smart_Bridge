from __future__ import annotations

from functools import lru_cache
import re

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

# ─────────────────────────────────────────────
#  EXPANDED MEDICINE DATABASE
# ─────────────────────────────────────────────
MED_DB = {
    "paracetamol": {
        "name": "Paracetamol",
        "standard_dose_mg": {"adult": 500, "child": 250},
        "description": "Pain reliever and fever reducer (antipyretic/analgesic).",
        "interactions": {"warfarin": "High | increases bleeding risk"},
        "side_effects": ["nausea", "rash", "liver damage (overdose)"],
        "category": "Analgesic"
    },
    "warfarin": {
        "name": "Warfarin",
        "standard_dose_mg": {"adult": 5, "child": None},
        "description": "Anticoagulant / blood thinner used to prevent clots.",
        "interactions": {
            "paracetamol": "High | increases bleeding risk",
            "aspirin": "High | severe bleeding risk",
            "ibuprofen": "High | increases anticoagulant effect"
        },
        "side_effects": ["bleeding", "bruising", "hair loss"],
        "category": "Anticoagulant"
    },
    "aspirin": {
        "name": "Aspirin",
        "standard_dose_mg": {"adult": 325, "child": None},
        "description": "Pain reliever, anti-inflammatory, and blood thinner.",
        "interactions": {
            "warfarin": "High | severe bleeding risk",
            "ibuprofen": "Moderate | reduces aspirin effectiveness"
        },
        "side_effects": ["stomach upset", "bleeding", "tinnitus"],
        "category": "NSAID / Antiplatelet"
    },
    "ibuprofen": {
        "name": "Ibuprofen",
        "standard_dose_mg": {"adult": 400, "child": 200},
        "description": "Nonsteroidal anti-inflammatory drug (NSAID) for pain and fever.",
        "interactions": {
            "aspirin": "Moderate | reduces aspirin effectiveness",
            "warfarin": "High | increases anticoagulant effect"
        },
        "side_effects": ["stomach pain", "heartburn", "kidney issues"],
        "category": "NSAID"
    },
    "amoxicillin": {
        "name": "Amoxicillin",
        "standard_dose_mg": {"adult": 500, "child": 250},
        "description": "Broad-spectrum antibiotic for bacterial infections.",
        "interactions": {
            "methotrexate": "High | increases methotrexate toxicity",
            "warfarin": "Moderate | may increase bleeding risk"
        },
        "side_effects": ["diarrhea", "rash", "nausea", "allergic reaction"],
        "category": "Antibiotic"
    },
    "metformin": {
        "name": "Metformin",
        "standard_dose_mg": {"adult": 500, "child": None},
        "description": "Oral diabetes medicine for Type 2 diabetes management.",
        "interactions": {
            "alcohol": "High | increases lactic acidosis risk",
            "contrast_dye": "High | kidney damage risk"
        },
        "side_effects": ["nausea", "diarrhea", "stomach upset", "lactic acidosis (rare)"],
        "category": "Antidiabetic"
    },
    "atorvastatin": {
        "name": "Atorvastatin",
        "standard_dose_mg": {"adult": 10, "child": None},
        "description": "Statin used to lower cholesterol and prevent cardiovascular disease.",
        "interactions": {
            "clarithromycin": "High | CYP3A4 inhibitor increases statin levels",
            "grapefruit": "High | avoid grapefruit juice"
        },
        "side_effects": ["muscle pain", "liver enzyme elevation", "headache"],
        "category": "Statin"
    },
    "cetirizine": {
        "name": "Cetirizine",
        "standard_dose_mg": {"adult": 10, "child": 5},
        "description": "Antihistamine for allergy relief (hay fever, hives).",
        "interactions": {
            "alcohol": "Moderate | increases drowsiness"
        },
        "side_effects": ["drowsiness", "dry mouth", "headache"],
        "category": "Antihistamine"
    },
    "omeprazole": {
        "name": "Omeprazole",
        "standard_dose_mg": {"adult": 20, "child": 10},
        "description": "Proton pump inhibitor (PPI) for acid reflux and ulcers.",
        "interactions": {
            "clopidogrel": "High | reduces clopidogrel effectiveness",
            "methotrexate": "Moderate | increases methotrexate levels"
        },
        "side_effects": ["headache", "diarrhea", "nausea", "vitamin B12 deficiency (long-term)"],
        "category": "PPI / Antacid"
    },
    "methotrexate": {
        "name": "Methotrexate",
        "standard_dose_mg": {"adult": 7.5, "child": None},
        "description": "Immunosuppressant used for cancer, psoriasis, and rheumatoid arthritis.",
        "interactions": {
            "amoxicillin": "High | increases methotrexate toxicity",
            "omeprazole": "Moderate | increases methotrexate levels",
            "aspirin": "High | increases methotrexate toxicity"
        },
        "side_effects": ["nausea", "mouth sores", "liver toxicity", "bone marrow suppression"],
        "category": "Immunosuppressant"
    },
    "lisinopril": {
        "name": "Lisinopril",
        "standard_dose_mg": {"adult": 10, "child": None},
        "description": "ACE inhibitor for high blood pressure and heart failure.",
        "interactions": {
            "potassium": "High | hyperkalemia risk",
            "spironolactone": "High | dangerous potassium increase"
        },
        "side_effects": ["dry cough", "dizziness", "high potassium", "kidney issues"],
        "category": "ACE Inhibitor"
    },
    "azithromycin": {
        "name": "Azithromycin",
        "standard_dose_mg": {"adult": 500, "child": 250},
        "description": "Macrolide antibiotic for respiratory and skin infections.",
        "interactions": {
            "warfarin": "Moderate | increases bleeding risk",
            "amiodarone": "High | increases risk of heart arrhythmia"
        },
        "side_effects": ["nausea", "diarrhea", "stomach pain", "heart rhythm changes"],
        "category": "Antibiotic"
    }
}

MED_KEYS = tuple(MED_DB.keys())


# ─────────────────────────────────────────────
#  FUZZY MEDICINE FINDER
# ─────────────────────────────────────────────
@lru_cache(maxsize=4096)
def _find_medicine_cached(query: str, threshold: int) -> str | None:
    if query in MED_DB:
        return query  # exact match

    result = process.extractOne(query, MED_KEYS, scorer=fuzz.WRatio)
    if result and result[1] >= threshold:
        return result[0]
    return None


def find_medicine(query: str, threshold: int = 75) -> str | None:
    """
    Finds the closest matching medicine name in MED_DB using fuzzy matching.
    Returns the matched key or None if no good match found.
    """
    if not isinstance(query, str):
        return None
    cleaned_query = query.lower().strip()
    if not cleaned_query:
        return None
    return _find_medicine_cached(cleaned_query, int(threshold))


# ─────────────────────────────────────────────
#  INTERACTION CHECKER (UPGRADED)
# ─────────────────────────────────────────────
def check_interactions(medications: list) -> list:
    """
    Checks for known drug-drug interactions between a list of medications.
    Returns a list of interaction warning strings.
    """
    interactions_found = []
    resolved = []
    seen_pairs = set()

    # First resolve all medicine names via fuzzy matching
    for med in medications:
        matched = find_medicine(med)
        if matched:
            if matched not in resolved:
                resolved.append(matched)

    # Cross-check each pair (order-independent)
    for i, med_a in enumerate(resolved):
        for med_b in resolved[i + 1:]:
            pair_key = tuple(sorted((med_a, med_b)))
            if pair_key in seen_pairs:
                continue

            interaction_ab = MED_DB[med_a].get("interactions", {}).get(med_b)
            interaction_ba = MED_DB[med_b].get("interactions", {}).get(med_a)
            severity = interaction_ab or interaction_ba

            if severity:
                interactions_found.append(
                    f"⚠️ {med_a.capitalize()} ↔ {med_b.capitalize()} | {severity}"
                )
                seen_pairs.add(pair_key)

    return interactions_found


# ─────────────────────────────────────────────
#  EXTRACT MEDICINES FROM OCR TEXT
# ─────────────────────────────────────────────
def extract_medicines_from_text(text: str) -> list:
    """
    Scans OCR-extracted text word by word to find medicine names
    using fuzzy matching against the database.
    """
    found = []
    seen = set()
    words = re.findall(r"[A-Za-z]{4,}", (text or "").lower())
    unique_words = list(dict.fromkeys(words))

    for word in unique_words:
        med = find_medicine(word)
        if med and med not in seen:
            found.append(med)
            seen.add(med)

    return found






# MED_DB = {
#     "paracetamol": {
#         "description": "Pain reliever and fever reducer.",
#         "interactions": ["warfarin"],
#         "side_effects": ["nausea", "rash"]
#     },
#     "warfarin": {
#         "description": "Blood thinner.",
#         "interactions": ["paracetamol", "aspirin"],
#         "side_effects": ["bleeding"]
#     },
#     "aspirin": {
#         "description": "Pain reliever and anti-inflammatory.",
#         "interactions": ["warfarin", "ibuprofen"],
#         "side_effects": ["stomach upset", "bleeding"]
#     },
#     "ibuprofen": {
#         "description": "Nonsteroidal anti-inflammatory drug (NSAID).",
#         "interactions": ["aspirin"],
#         "side_effects": ["stomach pain", "heartburn"]
#     }
# }

# def check_interactions(medications: list) -> list:
#     """Checks for interactions between a list of specified medications."""
#     interactions_found = []
    
#     for i, med_a in enumerate(medications):
#         med_a = med_a.lower()
#         if med_a in MED_DB:
#             for med_b in medications[i+1:]:
#                 med_b = med_b.lower()
#                 if med_b in MED_DB[med_a].get("interactions", []):
#                     interactions_found.append(f"Interaction between {med_a.capitalize()} and {med_b.capitalize()}")
    
#     return interactions_found
