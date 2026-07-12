"""NLP evaluation: parser accuracy + RAG retrieval hit-rate -> reports/nlp_eval.md."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

from config import PROJECT_ROOT, settings  # noqa: E402
from src.nlp.retriever import build_index, retrieve  # noqa: E402

# 10 hand-written queries with the guideline doc expected in top-4.
RAG_QUERIES: list[tuple[str, str]] = [
    ("gündə neçə kalori yeməliyəm?", "kalori_balansi.md"),
    ("arıqlamaq üçün nə edim?", "ariqlama_prinsipleri.md"),
    ("idmandan əvvəl nə yeyim?", "idman_oncesi_qidalanma.md"),
    ("zülal norması nə qədərdir?", "zulal_ehtiyaci.md"),
    ("duzlu yemək təzyiqə təsir edirmi?", "natrium_ve_teziq.md"),
    ("şəkərli içkilər niyə zərərlidir?", "seker_ve_qlikemik_indeks.md"),
    ("diabet xəstəsi nə yeməlidir?", "diabet_ucun_qeydler.md"),
    ("lifli qidalar hansılardır?", "lifli_qidalar.md"),
    ("gündə nə qədər su içmək lazımdır?", "su_ve_maye_qebulu.md"),
    ("pitsanın kalorisi nə qədərdir?", "nutrition:pizza"),
]


def rag_hit_rate(k: int = 4) -> tuple[int, list[str]]:
    build_index()
    hits, rows = 0, []
    for query, expected in RAG_QUERIES:
        chunks = retrieve(query, k=k)
        sources = [c.source for c in chunks]
        hit = expected in sources
        hits += hit
        rows.append(f"| {query} | `{expected}` | {'✅' if hit else '❌ ' + sources[0]} |")
    return hits, rows


def parser_results() -> str:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_nlp.py", "-q", "--tb=no"],
        capture_output=True, text=True, cwd=PROJECT_ROOT)
    return r.stdout.strip().splitlines()[-1]


def main() -> None:
    hits, rows = rag_hit_rate()
    parser_line = parser_results()
    md = f"""# NLP qiymətləndirməsi / NLP evaluation

## 1. Meal parser (NER)

`tests/test_nlp.py` 20 Azərbaycanca cümlə üzərində işləyir (kəmiyyət sözləri,
vahidlər, çoxlu yemək, mənfi hal, bilinməyən yemək):

```
{parser_line}
```

Qeyd (dürüstlük): spec-dəki 0.55 embedding həddi MiniLM-in Azərbaycan sözlərində
səthi oxşarlıq səhvləri verdiyi üçün 0.75-ə qaldırılıb. Ölçülmüş nümunə:
"qutab" 0.70 kosinus oxşarlığı ilə "guacamole"-yə uyğunlaşırdı (yanlış müsbət).
0.75 həddində bilinməyən yeməklər düzgün şəkildə "unknown" kimi qaytarılır.
Bunun müqabilində lüğətdənkənar bəzi düzgün uyğunlaşmalar itirilir; geniş
sinonim lüğəti bunu kompensasiya edir. MiniLM ingilis mərkəzli modeldir və
Azərbaycan dili üçün zəifdir; çoxdilli model (məs. paraphrase-multilingual)
gələcək təkmilləşdirmə kimi qeyd olunur.

## 2. RAG retrieval hit-rate

10 əl ilə yazılmış sorğu; gözlənilən mənbə top-4 nəticədə olmalıdır (k=4):

| Sorğu | Gözlənilən mənbə | Nəticə |
|---|---|---|
{chr(10).join(rows)}

**Hit-rate: {hits}/10 ({hits * 10}%)**

Retriever: `all-MiniLM-L6-v2`, numpy kosinus axtarışı, {len(RAG_QUERIES)} sorğu.
Korpus: 9 guideline sənədi (~200 sözlük parçalar, 40 söz üst-üstə düşmə) +
25 sintetik qida sənədi.
"""
    out = settings.reports_dir / "nlp_eval.md"
    out.write_text(md, encoding="utf-8")
    print(f"parser: {parser_line}")
    print(f"RAG hit-rate: {hits}/10")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
