from med_db import MED_DB, check_interactions

# ─────────────────────────────────────────────
#  RISK SCORING ENGINE (TRANSPARENT / PERCENTAGE)
# ─────────────────────────────────────────────

# Maximum possible points — used to normalize to percentage
MAX_POINTS = 40


def calculate_risk_score(age: int, severity: str,
                          medicines: list = None,
                          has_interactions: bool = False,
                          chronic_conditions: list = None) -> dict:
    """
    Calculates a transparent, rule-based emergency risk score.

    Parameters:
    - age: Patient age
    - severity: "Low" | "Medium" | "High" | "Critical"
    - medicines: List of medicine names being taken
    - has_interactions: Whether drug interactions were detected
    - chronic_conditions: List of conditions e.g. ["diabetes", "heart disease"]

    Returns a dict with:
    - score: raw points
    - percentage: 0-100%
    - category: Low / Medium / High / Critical
    - breakdown: dict showing point contributions
    """
    breakdown = {}
    total = 0

    medicines = medicines or []
    chronic_conditions = chronic_conditions or []

    # ── 1. Severity Score (max 16 points) ──
    severity_map = {
        "Low": 2,
        "Medium": 6,
        "High": 12,
        "Critical": 16
    }
    sev_pts = severity_map.get(severity, 0)
    breakdown["Symptom Severity"] = f"{sev_pts} pts ({severity})"
    total += sev_pts

    # ── 2. Age Risk Factor (max 6 points) ──
    if age < 2:
        age_pts = 6
        age_label = "Infant (<2)"
    elif age < 5:
        age_pts = 5
        age_label = "Toddler (2-4)"
    elif age < 12:
        age_pts = 2
        age_label = "Child (5-11)"
    elif age > 75:
        age_pts = 6
        age_label = "Elderly (>75)"
    elif age > 65:
        age_pts = 4
        age_label = "Senior (65-75)"
    else:
        age_pts = 0
        age_label = "Adult (12-65, normal risk)"
    breakdown["Age Risk"] = f"{age_pts} pts ({age_label})"
    total += age_pts

    # ── 3. Drug Interaction Risk (max 8 points) ──
    if medicines:
        interactions = check_interactions(medicines)
        num_interactions = len(interactions)
        interaction_pts = min(num_interactions * 4, 8)
    else:
        interaction_pts = 0
    breakdown["Drug Interactions"] = f"{interaction_pts} pts ({len(medicines)} meds checked)"
    total += interaction_pts

    # ── 4. High-Risk Medicine Penalty (max 4 points) ──
    HIGH_RISK_MEDS = ["warfarin", "methotrexate", "metformin"]
    hr_count = sum(1 for m in medicines if m.lower() in HIGH_RISK_MEDS)
    hr_pts = min(hr_count * 2, 4)
    breakdown["High-Risk Medicines"] = f"{hr_pts} pts ({hr_count} high-risk meds)"
    total += hr_pts

    # ── 5. Chronic Conditions (max 6 points) ──
    HIGH_RISK_CONDITIONS = ["diabetes", "heart disease", "kidney disease",
                            "liver disease", "hypertension", "cancer"]
    cond_count = sum(1 for c in chronic_conditions
                     if any(h in c.lower() for h in HIGH_RISK_CONDITIONS))
    cond_pts = min(cond_count * 2, 6)
    breakdown["Chronic Conditions"] = f"{cond_pts} pts ({cond_count} high-risk conditions)"
    total += cond_pts

    # ── Normalize to percentage ──
    percentage = round((total / MAX_POINTS) * 100, 1)
    percentage = min(percentage, 100.0)

    # ── Determine Category ──
    if percentage >= 75:
        category = "Critical"
    elif percentage >= 50:
        category = "High"
    elif percentage >= 25:
        category = "Medium"
    else:
        category = "Low"

    return {
        "score": total,
        "max_score": MAX_POINTS,
        "percentage": percentage,
        "category": category,
        "breakdown": breakdown
    }


# ─────────────────────────────────────────────
#  SAFETY RULES ENGINE
# ─────────────────────────────────────────────
SAFETY_RULES = [
    {
        "condition": lambda age, meds, severity: age < 16 and "aspirin" in [m.lower() for m in meds],
        "warning": "🚨 SAFETY ALERT: Aspirin is contraindicated in patients under 16 (Reye's syndrome risk)."
    },
    {
        "condition": lambda age, meds, severity: severity == "Critical",
        "warning": "🚨 CRITICAL SEVERITY: Seek emergency medical care immediately!"
    },
    {
        "condition": lambda age, meds, severity: age > 65 and "warfarin" in [m.lower() for m in meds],
        "warning": "⚠️ Warfarin in elderly patient: High bleeding risk. Regular INR monitoring required."
    },
    {
        "condition": lambda age, meds, severity: "methotrexate" in [m.lower() for m in meds],
        "warning": "⚠️ Methotrexate detected: Requires specialist supervision and regular blood tests."
    }
]


def apply_safety_rules(age: int, medicines: list, severity: str) -> list:
    """
    Applies predefined safety rules and returns triggered warnings.
    """
    triggered = []
    for rule in SAFETY_RULES:
        try:
            if rule["condition"](age, medicines, severity):
                triggered.append(rule["warning"])
        except Exception:
            continue
    return triggered








# def calculate_risk_score(age: int, severity: str) -> tuple:
#     """
#     Calculates an emergency risk score out of 10 and returns the category.
#     """
#     score = 0
    
#     # Base severity score
#     severity_scores = {
#         "Low": 1,
#         "Medium": 4,
#         "High": 7,
#         "Critical": 10
#     }
    
#     score += severity_scores.get(severity, 0)
    
#     # Age factor
#     if age < 5 or age > 65:
#         score += 2
    
#     # Cap score at 10
#     score = min(score, 10)
    
#     category = "Low"
#     if score >= 8:
#         category = "Critical"
#     elif score >= 6:
#         category = "High"
#     elif score >= 4:
#         category = "Medium"
        
#     return score, category
