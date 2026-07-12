"""RAG advisor: MealAnalysis + UserProfile -> Azerbaijani advice text.

All three LLM providers share one path: the RAG context is retrieved, then
either sent to a generative provider (anthropic/local) or rendered through a
deterministic Jinja template (default). Every output ends with the disclaimer.
"""

from __future__ import annotations

import logging

from jinja2 import Template

from src.nlp import llm
from src.nlp.retriever import retrieve
from src.schemas import DISCLAIMER_AZ, MealAnalysis, UserProfile

log = logging.getLogger(__name__)

SYSTEM_AZ = (
    "Sən qidalanma məsləhətçisisən. Aşağıdakı yeməyi və istifadəçi profilini "
    "nəzərə alaraq 3-4 cümləlik konkret, praktiki məsləhət ver. Yalnız verilən "
    "konteksti istifadə et. Tibbi diaqnoz qoyma. Cavabı Azərbaycan dilində ver."
)

# Concrete swap suggestions keyed by class (fallback: generic by tag).
SWAPS_AZ: dict[str, str] = {
    "pizza": "nazik xəmirli və tərəvəzli pitsa seçin",
    "hamburger": "qril toyuq burgeri və bol salatla əvəz edin",
    "french_fries": "sobada bişmiş kartofla əvəz edin",
    "donuts": "meyvə və bir ovuc qoz-fındıqla əvəz edin",
    "cheesecake": "kiçik dilim seçin və ya meyvəli desertə üstünlük verin",
    "ice_cream": "meyvəli sorbetlə əvəz edin",
    "hot_dog": "evdə hazırlanmış toyuq sendviçi ilə əvəz edin",
    "fried_rice": "buxarda bişmiş düyü və tərəvəzlə əvəz edin",
    "ramen": "bulyonun hamısını içməyin, tərəvəz əlavə edin",
    "waffles": "yulaf pankeki və meyvə ilə əvəz edin",
    "pancakes": "siropu meyvə ilə əvəz edin",
    "tacos": "qızardılmış yox, qril ət seçin",
    "club_sandwich": "tam taxıl çörək və az mayonezlə hazırladın",
    "lasagna": "porsiyanı kiçildin, yanında yaşıl salat yeyin",
}
GENERIC_SWAP_AZ = "porsiyanı kiçildib yanına tərəvəz salatı əlavə edin"

TEMPLATE_AZ = Template(
    "{{ az_name }} ({{ grams|int }} q) təxminən {{ kcal|int }} kkal verir, bu da "
    "gündəlik hədəfinizin ({{ target|int }} kkal) {{ pct }}%-i deməkdir. "
    "{% if pct > 40 %}Bu, bir yemək üçün yüksək paydır, günün qalan yeməklərini "
    "yüngül saxlayın. {% elif pct > 25 %}Bu, əsas yemək üçün normal paydır. "
    "{% else %}Bu, gündəlik büdcənizə rahat yerləşir. {% endif %}"
    "{% if 'high_sodium' in tags %}Diqqət: bu yeməkdə natrium yüksəkdir "
    "({{ sodium|int }} mq), gün ərzində əlavə duzlu qidalardan çəkinin və su için. "
    "{% endif %}"
    "{% if 'high_sugar' in tags %}Diqqət: şəkər miqdarı yüksəkdir "
    "({{ sugar|int }} q), bu gün əlavə şirniyyatdan uzaq durun. {% endif %}"
    "Alternativ olaraq {{ swap }}."
)


def _build_context(meal: MealAnalysis) -> str:
    query = f"{meal.az_name} {' '.join(meal.tags)} qidalanma məsləhəti"
    chunks = retrieve(query, k=4)
    return "\n---\n".join(f"[{c.source}] {c.text}" for c in chunks)


def advise(meal: MealAnalysis, profile: UserProfile) -> str:
    """Generate Azerbaijani advice for one analysed meal.

    Includes kcal vs daily target %, a concrete swap suggestion, and a
    warning when tags contain high_sodium/high_sugar. Always appends the
    not-medical-advice disclaimer.
    """
    target = profile.daily_kcal_target or 2000.0
    kcal = meal.macros.kcal if meal.macros else 0.0
    pct = round(100.0 * kcal / max(target, 1))
    swap = SWAPS_AZ.get(meal.food_class, GENERIC_SWAP_AZ)

    rendered = TEMPLATE_AZ.render(
        az_name=meal.az_name or meal.food_class, grams=meal.grams, kcal=kcal,
        target=target, pct=pct, tags=meal.tags, swap=swap,
        sodium=meal.macros.sodium_mg if meal.macros else 0,
        sugar=meal.macros.sugar_g if meal.macros else 0)

    if llm.active_provider() == "template":
        text = llm.generate(SYSTEM_AZ, rendered)
    else:
        context = _build_context(meal)
        user = (f"Kontekst:\n{context}\n\nYemək: {meal.az_name}, {meal.grams:.0f} q, "
                f"{kcal:.0f} kkal (gündəlik hədəfin {pct}%-i). "
                f"Etiketlər: {', '.join(meal.tags) or 'yoxdur'}. "
                f"Profil: {profile.age} yaş, məqsəd: {profile.goal}, "
                f"hədəf {target:.0f} kkal.\n"
                f"Məsləhətə mütləq daxil et: kalori payı, bir konkret əvəzləmə "
                f"({swap}), varsa duz/şəkər xəbərdarlığı.")
        text = llm.generate(SYSTEM_AZ, user)

    return f"{text}\n\n{DISCLAIMER_AZ}"


if __name__ == "__main__":
    import sys

    from src.schemas import Macros

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    demo = MealAnalysis(ok=True, food_class="pizza", az_name="Pitsa", confidence=0.93,
                        grams=320, tags=["fast_food", "high_sodium", "high_carb"],
                        macros=Macros(kcal=851, protein_g=35, carb_g=106, fat_g=32,
                                      sodium_mg=1914, sugar_g=12))
    print(advise(demo, UserProfile(daily_kcal_target=2000)))
