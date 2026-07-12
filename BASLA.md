# BAŞLA — FoodLens-i necə açıb test edim?

## Ən asan yol: Desktop-dakı **FoodLens** shortcut-u

Masaüstündə **FoodLens** ikonuna iki dəfə klik → hər şey özü işə düşür:
venv yoxlanılır (yoxdursa qurulur), nutrition DB yoxlanılır, **API açılır**
(<http://127.0.0.1:8000/docs>), **demo açılır** (<http://localhost:8501>).
Heç bir manual addım yoxdur. Demo pəncərəsini bağlayanda API də bağlanır.

Alternativlər:

| Nə istəyirsən | Nə et |
|---|---|
| **Hamısı birdən** | Desktop → `FoodLens` (və ya qovluqda `START.bat`) |
| **Yalnız demo** | `DEMO.bat` faylına iki dəfə klik |
| **Testləri işlət** | `TESTLER.bat` faylına iki dəfə klik |

## Demoda nəyi yoxla

1. **📷 Şəkil analizi** tabı: internetdən istənilən pitsa/burger şəkli yüklə →
   maska + Grad-CAM + qram + kalori + "Məsləhət al" düyməsi.
   Test şəkilləri hazır qovluqdadır: `data/raw/food-101/images/pizza/` (istənilən birini götür).
2. **✍️ Mətnlə əlavə et** tabı: yaz: `2 dilim pitsa və bir stəkan kola` → tanınan yeməklər çıxacaq.
3. **📊 Gündəlik** tabı: bir-iki yemək yazandan sonra "Günü yekunlaşdır" düyməsi.
4. **🧠 Model** tabı: dəqiqlik cədvəli, confusion matrix, Grad-CAM nümunələri.

## PowerShell əmrləri (istəsən)

```powershell
.\run.ps1 install        # ilk quraşdırma (bir dəfə)
.\run.ps1 data           # dataset + nutrition DB
.\run.ps1 train-effnet   # əsas modeli öyrət
.\run.ps1 evaluate       # metrikləri çıxar
.\run.ps1 demo           # Streamlit demo
.\run.ps1 test           # pytest
```

## Vacib

- Demo **internetsiz və API açarsız** işləyir (`LLM_PROVIDER=template` defoltdur) — müdafiə günü sığortan budur.
- Şəkil analizi yalnız model öyrədiləndən sonra işləyir (`models/effnet_best.pt` olmalıdır).
  Mətn tabı və gündəlik modeldən asılı deyil, dərhal işləyir.
