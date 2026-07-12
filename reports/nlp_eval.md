# NLP qiymətləndirməsi / NLP evaluation

## 1. Meal parser (NER)

`tests/test_nlp.py` 20 Azərbaycanca cümlə üzərində işləyir (kəmiyyət sözləri,
vahidlər, çoxlu yemək, mənfi hal, bilinməyən yemək):

```
20 passed in 11.89s
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
| gündə neçə kalori yeməliyəm? | `kalori_balansi.md` | ✅ |
| arıqlamaq üçün nə edim? | `ariqlama_prinsipleri.md` | ✅ |
| idmandan əvvəl nə yeyim? | `idman_oncesi_qidalanma.md` | ✅ |
| zülal norması nə qədərdir? | `zulal_ehtiyaci.md` | ✅ |
| duzlu yemək təzyiqə təsir edirmi? | `natrium_ve_teziq.md` | ✅ |
| şəkərli içkilər niyə zərərlidir? | `seker_ve_qlikemik_indeks.md` | ✅ |
| diabet xəstəsi nə yeməlidir? | `diabet_ucun_qeydler.md` | ✅ |
| lifli qidalar hansılardır? | `lifli_qidalar.md` | ❌ idman_oncesi_qidalanma.md |
| gündə nə qədər su içmək lazımdır? | `su_ve_maye_qebulu.md` | ✅ |
| pitsanın kalorisi nə qədərdir? | `nutrition:pizza` | ❌ seker_ve_qlikemik_indeks.md |

**Hit-rate: 8/10 (80%)**

Retriever: `all-MiniLM-L6-v2`, numpy kosinus axtarışı, 10 sorğu.
Korpus: 9 guideline sənədi (~200 sözlük parçalar, 40 söz üst-üstə düşmə) +
25 sintetik qida sənədi.
