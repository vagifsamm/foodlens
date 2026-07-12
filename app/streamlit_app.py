"""FoodLens Streamlit demo. All user-facing strings in Azerbaijani."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CLASSES, settings  # noqa: E402
from src import db  # noqa: E402
from src.nlp.meal_parser import parse_meal  # noqa: E402
from src.nlp.summarizer import daily_summary  # noqa: E402
from src.pipeline import analysis_extras, analyze, load_nutrition_db, macros_for  # noqa: E402
from src.schemas import UserProfile  # noqa: E402

st.set_page_config(page_title="FoodLens", page_icon="🍽️", layout="wide")

# ---------- styling ----------
st.markdown("""
<style>
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
    .stApp { background: #FDF8F3; }
    h1, h2, h3 { color: #4A2C1A !important; font-family: Georgia, serif; }
    section[data-testid="stSidebar"] {
        background: #2E1F16;
    }
    section[data-testid="stSidebar"] * { color: #F3E6D8; }
    section[data-testid="stSidebar"] .fl-card { background: #3D2B1F; border-color: #5A4232; }
    section[data-testid="stSidebar"] .fl-kpi { color: #F2A65A; }
    section[data-testid="stSidebar"] .fl-label { color: #C9AE94; }
    .fl-card {
        background: #FFFFFF; border-radius: 16px; padding: 1.2rem 1.5rem;
        border: 1px solid #EAD9C9; box-shadow: 0 2px 10px rgba(74,44,26,.06);
        margin-bottom: 1rem;
    }
    .fl-kpi { font-size: 2rem; font-weight: 700; color: #C2571B; }
    .fl-label { color: #8A6D57; font-size: .85rem; text-transform: uppercase;
                letter-spacing: .06em; }
    .fl-chip { display: inline-block; background: #F6E7D8; color: #7A4A21;
               border-radius: 999px; padding: .15rem .7rem; margin: .1rem;
               font-size: .8rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1.02rem; }
    div.stButton > button {
        background: #C2571B; color: white; border: none; border-radius: 10px;
        padding: .5rem 1.4rem;
    }
    div.stButton > button:hover { background: #A34614; color: white; }
</style>
""", unsafe_allow_html=True)


def effnet_available() -> bool:
    return (settings.models_dir / "effnet_best.pt").exists()


@st.cache_resource
def nutrition_db() -> dict:
    return load_nutrition_db()


# ---------- sidebar: profile ----------
with st.sidebar:
    st.title("🍽️ FoodLens")
    st.caption("Yemək fotosundan kaloriyə, oradan məsləhətə")
    st.subheader("Profil")
    name = st.text_input("Ad", "Qonaq")
    col_a, col_b = st.columns(2)
    age = col_a.number_input("Yaş", 10, 100, 30)
    sex = col_b.selectbox("Cins", ["m", "f"], format_func=lambda s: "Kişi" if s == "m" else "Qadın")
    height_cm = col_a.number_input("Boy (sm)", 100, 230, 175)
    weight_kg = col_b.number_input("Çəki (kq)", 30, 250, 75)
    activity = st.selectbox("Aktivlik", ["sedentary", "light", "moderate", "active", "very_active"],
                            index=1, format_func=lambda a: {
                                "sedentary": "Oturaq", "light": "Yüngül",
                                "moderate": "Orta", "active": "Aktiv",
                                "very_active": "Çox aktiv"}[a])
    goal = st.selectbox("Məqsəd", ["lose", "maintain", "gain"], index=1,
                        format_func=lambda g: {"lose": "Arıqlamaq",
                                               "maintain": "Saxlamaq",
                                               "gain": "Kütlə artırmaq"}[g])
    target = db.daily_kcal_target(age, sex, height_cm, weight_kg, activity, goal)
    st.markdown(f'<div class="fl-card"><div class="fl-label">Gündəlik hədəf</div>'
                f'<div class="fl-kpi">{target:.0f} kkal</div></div>',
                unsafe_allow_html=True)

profile = UserProfile(name=name, age=age, sex=sex, height_cm=height_cm,
                      weight_kg=weight_kg, activity=activity, goal=goal,
                      daily_kcal_target=float(target))


def get_user_id() -> int:
    session = db.get_session()
    user = db.get_or_create_user(session, name, age, sex, height_cm, weight_kg,
                                 activity, goal)
    return user.id


def log_meal(food_class: str, grams: float, confidence: float, source: str) -> None:
    macros, _, _ = macros_for(food_class, grams)
    session = db.get_session()
    row = db.MealLog(user_id=get_user_id(), food_class=food_class,
                     confidence=confidence, grams=grams, kcal=macros.kcal,
                     protein_g=macros.protein_g, carb_g=macros.carb_g,
                     fat_g=macros.fat_g, sodium_mg=macros.sodium_mg, source=source)
    session.add(row)
    session.commit()


tab1, tab2, tab3, tab4 = st.tabs(
    ["📷 Şəkil analizi", "✍️ Mətnlə əlavə et", "📊 Gündəlik", "🧠 Model"])

# ---------- tab 1: photo analysis ----------
with tab1:
    st.header("Şəkil analizi")
    if not effnet_available():
        st.warning("Model hələ öyrədilməyib. Əvvəlcə `.\\run.ps1 train-effnet` işlədin.")
    src_choice = st.radio("Mənbə", ["Fayl yüklə", "Kamera"], horizontal=True,
                          label_visibility="collapsed")
    file = (st.file_uploader("Yemək şəkli seçin",
                             type=["jpg", "jpeg", "png", "webp", "bmp"])
            if src_choice == "Fayl yüklə" else st.camera_input("Şəkil çəkin"))

    if file is not None and effnet_available():
        img = cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st.error("Bu şəkil formatı oxuna bilmədi. JPG və ya PNG yükləyin.")
            st.stop()
        with st.spinner("Analiz gedir..."):
            meal = analyze(img, profile, with_advice=False)
        if not meal.ok:
            for r in meal.quality_reasons_az:
                st.error(r)
        else:
            with st.spinner("Vizual nəticələr hazırlanır..."):
                extras = analysis_extras(img)
            c1, c2, c3, c4 = st.columns(4)
            c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Orijinal",
                     use_container_width=True)
            c2.image(extras["mask"], caption="Seqmentasiya maskası", clamp=True,
                     use_container_width=True)
            c3.image(cv2.cvtColor(extras["gradcam"], cv2.COLOR_BGR2RGB),
                     caption="Grad-CAM", use_container_width=True)
            with c4:
                st.markdown(f"""<div class="fl-card">
                    <div class="fl-label">Nəticə</div>
                    <div class="fl-kpi">{meal.az_name}</div>
                    <p>Əminlik: <b>{meal.confidence:.0%}</b><br>
                    Porsiya: <b>{meal.portion_bucket}</b>
                    ({'plitə tapıldı' if meal.portion_confidence == 'high' else 'boşqab tapılmadı, təxmini'})</p>
                    {''.join(f'<span class="fl-chip">{t}</span>' for t in meal.tags)}
                    </div>""", unsafe_allow_html=True)

            grams = st.slider("Porsiya (qram), lazımsa düzəldin", 20, 1500,
                              int(meal.grams), 10)
            macros, _, _ = macros_for(meal.food_class, grams)
            k1, k2, k3, k4 = st.columns(4)
            for col, label, val in ((k1, "Kalori", f"{macros.kcal:.0f} kkal"),
                                    (k2, "Zülal", f"{macros.protein_g:.0f} q"),
                                    (k3, "Karbohidrat", f"{macros.carb_g:.0f} q"),
                                    (k4, "Yağ", f"{macros.fat_g:.0f} q")):
                col.markdown(f'<div class="fl-card"><div class="fl-label">{label}</div>'
                             f'<div class="fl-kpi">{val}</div></div>',
                             unsafe_allow_html=True)

            b1, b2 = st.columns(2)
            if b1.button("💬 Məsləhət al", use_container_width=True):
                from src.nlp.advisor import advise

                meal.grams = float(grams)
                meal.macros = macros
                with st.spinner("Məsləhət hazırlanır..."):
                    st.info(advise(meal, profile))
            if b2.button("📒 Gündəliyə yaz", use_container_width=True):
                log_meal(meal.food_class, float(grams), meal.confidence, "photo")
                st.success("Yemək gündəliyə əlavə olundu.")

# ---------- tab 2: text entry ----------
with tab2:
    st.header("Mətnlə əlavə et")
    text = st.text_input("Nə yemisiniz?",
                         placeholder="məs: 2 dilim pitsa və bir stəkan kola")
    if text:
        entities = parse_meal(text)
        if not entities:
            st.info("Heç bir yemək tapılmadı.")
        db_json = nutrition_db()
        for i, e in enumerate(entities):
            with st.container():
                cols = st.columns([2, 1, 1, 1, 1])
                if e.in_db:
                    az = db_json[e.food]["az_name"]
                    cols[0].markdown(f'<span class="fl-chip">✅ {az}</span>',
                                     unsafe_allow_html=True)
                    default_g = float(db_json[e.food]["typical_serving_g"]) * e.qty
                    grams = cols[1].number_input("qram", 10.0, 3000.0, default_g,
                                                 key=f"g{i}")
                    m, _, _ = macros_for(e.food, grams)
                    cols[2].metric("kkal", f"{m.kcal:.0f}")
                    cols[3].write(f"{e.qty:g} {e.unit or 'porsiya'}")
                    if cols[4].button("Yaz", key=f"log{i}"):
                        log_meal(e.food, grams, 1.0, "text")
                        st.success(f"{az} gündəliyə yazıldı.")
                else:
                    cols[0].markdown(f'<span class="fl-chip">❓ {e.raw}</span>',
                                     unsafe_allow_html=True)
                    cols[1].caption("Bazada yoxdur, kalori hesablanmadı")

# ---------- tab 3: daily log ----------
with tab3:
    st.header("Gündəlik")
    session = db.get_session()
    logs = db.logs_for_date(session, get_user_id(), dt.date.today())
    if not logs:
        st.info("Bu gün üçün qeyd yoxdur. Şəkil və ya mətnlə yemək əlavə edin.")
    else:
        total_kcal = sum(r.kcal for r in logs)
        st.progress(min(total_kcal / target, 1.0),
                    text=f"{total_kcal:.0f} / {target:.0f} kkal")
        db_json = nutrition_db()
        for r in logs:
            az = db_json.get(r.food_class, {}).get("az_name", r.food_class)
            st.markdown(f'<div class="fl-card">🍴 <b>{az}</b>, {r.grams:.0f} q, '
                        f'{r.kcal:.0f} kkal <span class="fl-label">'
                        f'({r.ts:%H:%M}, {"şəkil" if r.source == "photo" else "mətn"})'
                        f'</span></div>', unsafe_allow_html=True)

        import matplotlib.pyplot as plt

        p = sum(r.protein_g for r in logs) * 4
        c = sum(r.carb_g for r in logs) * 4
        f = sum(r.fat_g for r in logs) * 9
        if p + c + f > 0:
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            ax.pie([p, c, f], labels=["Zülal", "Karbohidrat", "Yağ"],
                   colors=["#8093f1", "#f3c178", "#e07b39"], autopct="%1.0f%%",
                   wedgeprops={"width": 0.42})
            fig.patch.set_alpha(0)
            st.pyplot(fig, use_container_width=False)

        if st.button("📋 Günü yekunlaşdır"):
            with st.spinner("Xülasə hazırlanır..."):
                s = daily_summary(logs, profile, date=str(dt.date.today()))
            st.info(s.summary_az)
            st.success(s.plan_az)

# ---------- tab 4: model metrics ----------
with tab4:
    st.header("Model nəticələri")
    metrics_path = settings.reports_dir / "metrics.json"
    if not metrics_path.exists():
        st.info("Hələ qiymətləndirmə aparılmayıb: `.\\run.ps1 evaluate`")
    else:
        import json

        with metrics_path.open(encoding="utf-8") as fjson:
            metrics = json.load(fjson)
        rows = [{"Model": ("SimpleCNN (sıfırdan)" if k == "simple"
                           else "EfficientNet-B0 (transfer)"),
                 "Top-1": f"{v['top1_acc']:.1%}", "Top-5": f"{v['top5_acc']:.1%}",
                 "Macro F1": f"{v['macro_f1']:.3f}",
                 "Parametrlər": f"{v['params']:,}",
                 "CPU ms/şəkil": v["cpu_ms_per_image"]}
                for k, v in metrics.items()]
        st.table(rows)
        for img_name, caption in [
                ("model_comparison.png", "Model müqayisəsi"),
                ("confusion_matrix_effnet.png", "Confusion matrix (EfficientNet)"),
                ("confusion_matrix_simple.png", "Confusion matrix (SimpleCNN)")]:
            path = settings.reports_dir / img_name
            if path.exists():
                st.image(str(path), caption=caption)
        gradcam_dir = settings.reports_dir / "gradcam"
        if gradcam_dir.exists():
            st.subheader("Grad-CAM nümunələri")
            imgs = sorted(gradcam_dir.glob("*.png"))[:8]
            for row_start in range(0, len(imgs), 4):
                for col, p in zip(st.columns(4), imgs[row_start:row_start + 4]):
                    col.image(str(p), caption=p.stem[:40], use_container_width=True)
