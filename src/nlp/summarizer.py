"""Daily summarizer: meal logs -> Azerbaijani summary + next-day 3-meal plan."""

from __future__ import annotations

import json
import logging
from typing import Sequence

from jinja2 import Template

from config import settings
from src.nlp import llm
from src.schemas import DISCLAIMER_AZ, DailySummary, Macros, UserProfile

log = logging.getLogger(__name__)

SYSTEM_AZ = (
    "Sən qidalanma məsləhətçisisən. Günün yemək qeydlərinə əsasən qısa "
    "Azərbaycanca xülasə yaz və sabah üçün 3 yeməklik plan təklif et. "
    "Tibbi diaqnoz qoyma."
)

SUMMARY_TEMPLATE_AZ = Template(
    "Bu gün cəmi {{ kcal|int }} kkal qəbul etmisiniz, hədəfiniz {{ target|int }} "
    "kkal idi ({{ pct }}%). Makrolar: {{ protein|int }} q zülal, {{ carb|int }} q "
    "karbohidrat, {{ fat|int }} q yağ. "
    "{% if pct > 110 %}Hədəfi {{ (kcal - target)|int }} kkal aşmısınız, sabah "
    "porsiyaları bir az kiçiltməyə çalışın. "
    "{% elif pct < 70 %}Hədəfdən xeyli aşağıda qalmısınız, bədəninizin kifayət "
    "qədər enerji aldığından əmin olun. "
    "{% else %}Kalori balansınız yaxşı səviyyədədir. {% endif %}"
    "{% if protein_low %}Zülal qəbulunuz azdır, sabah zülallı qidalara yer verin. {% endif %}"
    "{% if sodium_high %}Bu gün natrium yüksək olub, sabah duzlu qidaları azaldın. {% endif %}"
)

PLAN_TEMPLATE_AZ = Template(
    "Sabah üçün plan (təxminən {{ total|int }} kkal):\n"
    "{% for m in meals %}• {{ m.slot }}: {{ m.az_name }}, {{ m.grams|int }} q "
    "(~{{ m.kcal|int }} kkal)\n{% endfor %}"
)

# Deterministic next-day plan candidates per slot, ordered by preference for
# each goal. Values are class keys from nutrition_db.
PLAN_CANDIDATES = {
    "lose": [("Səhər yeməyi", "omelette"), ("Nahar", "grilled_salmon"),
             ("Şam yeməyi", "greek_salad")],
    "maintain": [("Səhər yeməyi", "pancakes"), ("Nahar", "chicken_curry"),
                 ("Şam yeməyi", "greek_salad")],
    "gain": [("Səhər yeməyi", "omelette"), ("Nahar", "steak"),
             ("Şam yeməyi", "spaghetti_bolognese")],
}


def _load_db() -> dict:
    with settings.nutrition_db_path.open(encoding="utf-8") as f:
        return json.load(f)


def _totals(logs: Sequence) -> Macros:
    """Sum macros over MealLog-like rows (need kcal/protein_g/carb_g/fat_g)."""
    t = Macros(0, 0, 0, 0)
    for row in logs:
        t.kcal += float(getattr(row, "kcal", 0) or 0)
        t.protein_g += float(getattr(row, "protein_g", 0) or 0)
        t.carb_g += float(getattr(row, "carb_g", 0) or 0)
        t.fat_g += float(getattr(row, "fat_g", 0) or 0)
        t.sodium_mg += float(getattr(row, "sodium_mg", 0) or 0)
    return t


def _build_plan(profile: UserProfile, target: float) -> str:
    """Pick a 3-meal next-day plan within the kcal budget."""
    db = _load_db()
    meals = []
    budget = target
    for slot, cls in PLAN_CANDIDATES.get(profile.goal, PLAN_CANDIDATES["maintain"]):
        entry = db[cls]
        grams = float(entry["typical_serving_g"])
        kcal = grams / 100.0 * float(entry["per_100g"]["kcal"])
        # Scale the serving down if it would blow the remaining budget.
        remaining_slots = 3 - len(meals)
        cap = budget / max(remaining_slots, 1) * 1.2
        if kcal > cap:
            scale = cap / kcal
            grams, kcal = grams * scale, cap
        meals.append({"slot": slot, "az_name": entry["az_name"],
                      "grams": grams, "kcal": kcal})
        budget -= kcal
    total = sum(m["kcal"] for m in meals)
    return PLAN_TEMPLATE_AZ.render(meals=meals, total=total)


def daily_summary(logs: Sequence, profile: UserProfile,
                  date: str = "") -> DailySummary:
    """Roll up a day's MealLogs into an Azerbaijani summary + next-day plan."""
    target = profile.daily_kcal_target or 2000.0
    total = _totals(logs)
    pct = round(100.0 * total.kcal / max(target, 1))
    protein_target = profile.weight_kg * 0.8

    rendered = SUMMARY_TEMPLATE_AZ.render(
        kcal=total.kcal, target=target, pct=pct, protein=total.protein_g,
        carb=total.carb_g, fat=total.fat_g,
        protein_low=total.protein_g < protein_target * 0.7,
        sodium_high=total.sodium_mg > 2300)
    plan = _build_plan(profile, target)

    if llm.active_provider() == "template":
        summary_text = llm.generate(SYSTEM_AZ, rendered)
    else:
        user = (f"Gün ərzində qəbul: {total.kcal:.0f} kkal (hədəf {target:.0f}), "
                f"zülal {total.protein_g:.0f} q, karbohidrat {total.carb_g:.0f} q, "
                f"yağ {total.fat_g:.0f} q. Qeydlər: "
                + "; ".join(f"{getattr(r, 'food_class', '?')} {getattr(r, 'grams', 0):.0f} q"
                            for r in logs)
                + ". Qısa xülasə yaz (3-4 cümlə), Azərbaycan dilində.")
        summary_text = llm.generate(SYSTEM_AZ, user)

    return DailySummary(date=date, total=total, target_kcal=target,
                        summary_az=f"{summary_text}\n\n{DISCLAIMER_AZ}",
                        plan_az=plan)


if __name__ == "__main__":
    import sys
    from types import SimpleNamespace

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    fake_logs = [SimpleNamespace(food_class="pizza", grams=320, kcal=851,
                                 protein_g=35, carb_g=106, fat_g=32, sodium_mg=1914),
                 SimpleNamespace(food_class="ice_cream", grams=132, kcal=273,
                                 protein_g=5, carb_g=32, fat_g=15, sodium_mg=106)]
    s = daily_summary(fake_logs, UserProfile(daily_kcal_target=2000, goal="lose"))
    print(s.summary_az)
    print(s.plan_az)
