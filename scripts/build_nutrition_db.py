"""Generate data/nutrition_db.json for all 25 classes.

Values are hand-authored averages based on USDA FoodData Central entries for
typical restaurant preparations. Azerbaijani names and notes are reviewed by a
native speaker (STOP 3). Sources cited in README.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

from config import CLASSES, settings  # noqa: E402


def entry(az_name: str, kcal: float, protein: float, carb: float, fat: float,
          fiber: float, sugar: float, sodium: int, serving_g: int,
          serving_az: str, density: float, tags: list[str], notes_az: str) -> dict:
    """Build one nutrition_db entry in the spec 3.2 schema."""
    return {
        "az_name": az_name,
        "per_100g": {"kcal": kcal, "protein_g": protein, "carb_g": carb,
                     "fat_g": fat, "fiber_g": fiber, "sugar_g": sugar,
                     "sodium_mg": sodium},
        "typical_serving_g": serving_g,
        "serving_name_az": serving_az,
        "density_g_per_cm2": density,
        "tags": tags,
        "notes_az": notes_az,
    }


DB: dict[str, dict] = {
    "pizza": entry("Pitsa", 266, 11.0, 33.0, 10.0, 2.3, 3.6, 598, 107, "dilim", 1.1,
                   ["fast_food", "high_sodium", "high_carb"],
                   "Yüksək natrium və doymuş yağ var. Nazik xəmirli və tərəvəzli variant daha yaxşı seçimdir."),
    "hamburger": entry("Hamburger", 254, 13.0, 24.0, 12.0, 1.5, 5.0, 414, 226, "ədəd", 1.5,
                       ["fast_food", "high_fat", "high_sodium"],
                       "Kalori sıxlığı yüksəkdir. Sousu azaldıb salat əlavə etsəniz, daha balanslı olar."),
    "french_fries": entry("Kartof fri", 312, 3.4, 41.0, 15.0, 3.8, 0.3, 210, 117, "porsiya", 0.9,
                          ["fast_food", "high_fat", "high_carb"],
                          "Dərin yağda qızardılır və kalorisi yüksəkdir. Sobada bişmiş kartof daha sağlam alternativdir."),
    "caesar_salad": entry("Sezar salatı", 190, 8.0, 8.0, 14.0, 1.8, 2.0, 380, 220, "boşqab", 0.5,
                          ["high_fat", "high_sodium"],
                          "Kalorinin çoxu sousdan gəlir. Sousu ayrıca istəyib az istifadə etmək olar."),
    "sushi": entry("Suşi", 150, 6.0, 28.0, 1.5, 0.9, 3.0, 335, 200, "porsiya", 0.8,
                   ["high_carb", "high_sodium"],
                   "Yağı azdır, amma soya sousu natriumu xeyli artırır. Sousu az işlədin."),
    "steak": entry("Mal əti steyki", 271, 26.0, 0.0, 18.0, 0.0, 0.0, 58, 220, "porsiya", 1.3,
                   ["high_protein", "high_fat"],
                   "Zülalla zəngindir. Yağsız hissələri seçib tərəvəzlə birlikdə yemək məsləhətdir."),
    "fried_rice": entry("Qızardılmış düyü", 163, 4.5, 28.0, 4.0, 0.9, 1.0, 396, 198, "boşqab", 0.9,
                        ["high_carb", "high_sodium"],
                        "Adi düyüdən daha yağlı və duzludur. Tərəvəz və zülal əlavə etsəniz, daha doyumlu olar."),
    "spaghetti_bolognese": entry("Bolonez spagetti", 129, 7.0, 15.0, 4.5, 1.5, 2.5, 240, 300, "boşqab", 0.9,
                                 ["high_carb"],
                                 "Balanslı yeməkdir, amma porsiya ölçüsünə fikir verin. Tam taxıl makaron daha çox lif verir."),
    "pancakes": entry("Pankek", 227, 6.0, 28.0, 10.0, 1.2, 8.0, 430, 116, "ədəd", 0.8,
                      ["high_carb", "high_sugar"],
                      "Sirop və kərə yağı ilə kalorisi sürətlə artır. Meyvə ilə yemək daha yaxşı seçimdir."),
    "omelette": entry("Omlet", 154, 11.0, 1.0, 12.0, 0.0, 0.5, 155, 122, "porsiya", 0.9,
                      ["high_protein"],
                      "Zülalla zəngin səhər yeməyidir. Tərəvəz əlavə etsəniz, vitamin dəyəri artar."),
    "grilled_salmon": entry("Qril qızılbalıq", 208, 22.0, 0.0, 13.0, 0.0, 0.0, 59, 170, "porsiya", 1.0,
                            ["high_protein", "healthy_fat"],
                            "Omeqa-3 yağ turşuları ilə zəngindir. Həftədə 2 dəfə balıq yemək ürək sağlamlığına xeyirlidir."),
    "chicken_curry": entry("Toyuq kari", 120, 12.0, 6.0, 5.5, 1.0, 2.0, 350, 235, "porsiya", 0.9,
                           ["high_protein"],
                           "Zülal mənbəyidir. Kremli variantlar daha kalorili olur, pomidor əsaslı sous daha yüngüldür."),
    "donuts": entry("Ponçik", 421, 5.0, 50.0, 23.0, 1.5, 23.0, 373, 60, "ədəd", 0.7,
                    ["high_sugar", "high_fat", "fast_food"],
                    "Şəkər və yağ nisbəti çox yüksəkdir. Gündəlik yox, arabir yemək məsləhətdir."),
    "cheesecake": entry("Çizkeyk", 321, 5.5, 26.0, 23.0, 0.4, 22.0, 438, 125, "dilim", 1.0,
                        ["high_sugar", "high_fat"],
                        "Kalorili desertdir. Kiçik dilimlə kifayətlənmək və şəkərli içki ilə yeməmək daha yaxşıdır."),
    "ice_cream": entry("Dondurma", 207, 3.5, 24.0, 11.0, 0.7, 21.0, 80, 66, "kürəcik", 0.7,
                       ["high_sugar"],
                       "1 kürəcik ağlabatan porsiyadır. Meyvəli sorbet daha az kalorili alternativdir."),
    "hot_dog": entry("Hot-doq", 290, 10.0, 24.0, 17.0, 0.9, 4.0, 810, 150, "ədəd", 1.0,
                     ["fast_food", "high_sodium"],
                     "Natrium miqdarı çox yüksəkdir. Emal olunmuş ət məhsullarını tez-tez yemək məsləhət deyil."),
    "dumplings": entry("Düşbərə (dumplinq)", 190, 8.0, 25.0, 6.0, 1.2, 1.0, 420, 180, "porsiya", 0.9,
                       ["high_carb"],
                       "Buxarda bişmiş variant qızardılmışdan daha yüngüldür. Sirkə və istiotla dadlandırmaq olar."),
    "falafel": entry("Falafel", 333, 13.0, 32.0, 18.0, 4.9, 1.0, 294, 102, "porsiya", 0.8,
                     ["vegetarian", "high_fiber", "high_fat"],
                     "Bitki mənşəli zülal və lif mənbəyidir. Dərin yağda qızardıldığı üçün kalorisi yüksəkdir."),
    "greek_salad": entry("Yunan salatı", 105, 2.5, 6.0, 8.0, 1.9, 4.0, 310, 250, "boşqab", 0.5,
                         ["healthy_fat"],
                         "Zeytun yağı və tərəvəzlərlə yüngül seçimdir. Pendirin miqdarı duz balansına təsir edir."),
    "lasagna": entry("Lazanya", 155, 9.0, 14.0, 7.0, 1.3, 3.5, 340, 250, "dilim", 1.1,
                     ["high_carb", "high_sodium"],
                     "Doyurucu yeməkdir, amma porsiyanı böyütməyin. Yanında yaşıl salat yaxşı gedir."),
    "ramen": entry("Ramen şorbası", 89, 4.0, 12.0, 3.0, 0.8, 1.0, 450, 500, "kasa", 0.6,
                   ["high_sodium"],
                   "Bulyonun duzu çox olur. Bulyonu tam içməsəniz, natrium qəbulunuz xeyli azalar."),
    "waffles": entry("Vafli", 291, 7.9, 37.0, 13.0, 1.7, 10.0, 511, 75, "ədəd", 0.8,
                     ["high_carb", "high_sugar"],
                     "Sirop əlavə etməzdən əvvəl kalorini nəzərə alın. Meyvə və qatıqla daha balanslı olur."),
    "tacos": entry("Tako", 216, 12.0, 20.0, 10.0, 2.5, 2.0, 397, 170, "ədəd", 0.9,
                   ["fast_food", "high_sodium"],
                   "Tərkib seçimi kalorini müəyyən edir. Qızardılmış yox, qril ət və bol tərəvəz seçin."),
    "guacamole": entry("Quakamole", 157, 2.0, 8.5, 14.0, 6.7, 0.7, 315, 100, "porsiya", 0.7,
                       ["healthy_fat", "high_fiber"],
                       "Avokadonun faydalı yağları ilə zəngindir. Çips əvəzinə tərəvəz çubuqları ilə yeyin."),
    "club_sandwich": entry("Klub sendviç", 240, 14.0, 25.0, 9.5, 1.6, 4.0, 560, 230, "ədəd", 0.9,
                           ["high_sodium"],
                           "Mayonez və bekon kalorini artırır. Tam taxıl çörəklə və az sousla daha yaxşı olar."),
}


def main() -> None:
    missing = [c for c in CLASSES if c not in DB]
    extra = [c for c in DB if c not in CLASSES]
    if missing or extra:
        raise ValueError(f"DB mismatch: missing={missing}, extra={extra}")
    out = settings.nutrition_db_path
    with out.open("w", encoding="utf-8") as f:
        json.dump(DB, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out} with {len(DB)} entries")


if __name__ == "__main__":
    main()
