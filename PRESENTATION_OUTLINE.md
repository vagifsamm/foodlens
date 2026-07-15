# FoodLens — Təqdimat planı (12 slayd)

Müdafiə üçün 12 slayd, hər birində bir sətir spiker qeydi. Bütün mətn Azərbaycan dilində.

---

## Slayd 1 — Başlıq
**FoodLens: şəkildən qidaya**
Yemək şəkli → sinif + porsiya + kalori + Azərbaycan dilində məsləhət.
🎤 *Salam, mən FoodLens-i təqdim edirəm — telefonla çəkilmiş yemək şəklini oflayn analiz edən sistem.*

## Slayd 2 — Problem
İnsanlar nə yediyini qeyd etmək istəyir, amma kalori saymaq yorucu və qeyri-dəqiqdir.
🎤 *Məqsəd: bir şəkil çək — qalanını sistem etsin, özü də Azərbaycan dilində.*

## Slayd 3 — Həll: 4 real ML qatı
CV → CNN → NLP → Məsləhətçi. Heç bir qat API qısayolu ilə əvəz edilməyib.
🎤 *Qiymətin şərti budur ki, kompüter görməsi, CNN və NLP həqiqi olsun — biz məhz onu qurduq.*

## Slayd 4 — Arxitektura (diaqram)
`README.md`-dəki Mermaid axını: şəkil → keyfiyyət → boşqab → seqment → CNN + porsiya → qida bazası → RAG → məsləhət.
🎤 *Bütün konveyer bir `pipeline.py`-də birləşir; hər addım müstəqil test olunub.*

## Slayd 5 — Kompüter görməsi (OpenCV)
Keyfiyyət yoxlaması (Laplacian bulanıqlıq + HSV işıq) → HoughCircles boşqab → GrabCut maska → sahədən qram.
🎤 *Klassik CV — dərin şəbəkə deyil; şəffaf və izah edilə biləndir.*

## Slayd 6 — CNN: iki model
SimpleCNN (sıfırdan, 396K parametr) vs EfficientNet-B0 (transfer, 4M parametr). İki fazalı öyrətmə, erkən dayanma.
🎤 *Baseline-ı bilərəkdən sıfırdan yazdıq ki, transfer learning-in faydasını rəqəmlə göstərək.*

## Slayd 7 — Nəticələr (cədvəl)
| Model | Top-1 | Macro-F1 |
|-------|------:|---------:|
| SimpleCNN | 0.504 | 0.486 |
| EfficientNet-B0 | **0.923** | **0.922** |
🎤 *Transfer learning baseline-ı iki dəfədən çox üstələyir — gözlənilən, sağlam nəticə.*

## Slayd 8 — Grad-CAM (şəffaflıq)
Model şəkildə hara baxır? Düzgün və səhv nümunələr (məs. cheesecake→steak) istilik xəritəsi ilə.
🎤 *Səhv halları da göstəririk — model qablaşdırmaya deyil, yeməyə baxdığını isbat edirik.*

## Slayd 9 — NLP: parser + RAG
Azərbaycan dilində yemək NER (leksik sinonim + embedding) + hibrid RAG (kosinus + leksik). Parser 20/20, RAG 8/10.
🎤 *MiniLM Azərbaycan sözlərində səhv edirdi — leksik prefiks uyğunluğu ilə hit-rate-i qaldırdıq.*

## Slayd 10 — Porsiya: dürüst məhdudiyyət
MAE ≈ 159 q, MAPE ≈ 131 %. Tək şəkildən monokulyar kütlə — yaxşı-müəyyən problem deyil; hər çıxış `confidence` daşıyır.
🎤 *Bunu gizlətmirik — ölçdük, izah etdik; ən zəif hissə budur və biz bunu bilirik.*

## Slayd 11 — Demo
Streamlit: 4 tab (Şəkil / Mətn / Gündəlik / Model). Oflayn, API açarsız işləyir. Bir klik `START.bat`.
🎤 *İndi canlı göstərəcəyəm: pitsa şəkli yükləyirəm → maska, Grad-CAM, qram, kalori, məsləhət.*

## Slayd 12 — Etika, məhdudiyyət, gələcək iş
Tibbi məsləhət deyil (hər çıxışda disclaimer). Food-101 Qərb qərəzi. Gələcək: Azərbaycan mətbəxi sinifləri, çox-yeməkli boşqab, dərinlik.
🎤 *Təşəkkür edirəm — suallara hazıram.*

---

### Ehtiyat suallara cavablar
- **Niyə EfficientNet-B0?** Kiçik (4M), CPU-da 22 ms, 1660 Ti-də sığır, transfer üçün ideal.
- **Niyə AMP yox?** GTX 16xx-də BatchNorm-u NaN edir (DECISIONS.md #11) — sürət qazancı da yoxdur (tensor core yoxdur).
- **RAG-da niyə vektor bazası yox?** 25 sinif üçün numpy kosinus indeksi kifayətdir — sadə, asılılıqsız, reproduktiv.
