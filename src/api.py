"""FastAPI endpoints (spec section 7)."""

from __future__ import annotations

import datetime as dt
import logging
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src import db
from src.nlp.advisor import advise
from src.nlp.meal_parser import parse_meal
from src.nlp.summarizer import daily_summary
from src.pipeline import analyze, macros_for
from src.schemas import MealAnalysis, UserProfile

log = logging.getLogger(__name__)

app = FastAPI(title="FoodLens API", version="1.0.0",
              description="Yemək fotosu -> sinif + porsiya + qidalanma + məsləhət")


class MacrosOut(BaseModel):
    kcal: float
    protein_g: float
    carb_g: float
    fat_g: float
    fiber_g: float = 0.0
    sugar_g: float = 0.0
    sodium_mg: float = 0.0


class AnalysisOut(BaseModel):
    ok: bool
    food_class: str = ""
    az_name: str = ""
    confidence: float = 0.0
    top5: list[tuple[str, float]] = []
    grams: float = 0.0
    portion_bucket: str = ""
    portion_confidence: str = ""
    macros: Optional[MacrosOut] = None
    tags: list[str] = []
    quality_reasons_az: list[str] = []
    advice_az: str = ""


class ProfileIn(BaseModel):
    name: str = "Qonaq"
    age: int = Field(30, ge=10, le=100)
    sex: str = Field("m", pattern="^[mf]$")
    height_cm: float = Field(175, ge=100, le=230)
    weight_kg: float = Field(75, ge=30, le=250)
    activity: str = "light"
    goal: str = "maintain"

    def to_profile(self) -> UserProfile:
        target = db.daily_kcal_target(self.age, self.sex, self.height_cm,
                                      self.weight_kg, self.activity, self.goal)
        return UserProfile(name=self.name, age=self.age, sex=self.sex,
                           height_cm=self.height_cm, weight_kg=self.weight_kg,
                           activity=self.activity, goal=self.goal,
                           daily_kcal_target=target)


class ParseIn(BaseModel):
    text: str


class EntityOut(BaseModel):
    food: str
    qty: float
    unit: Optional[str]
    raw: str
    in_db: bool
    matched_via: str


class LogIn(BaseModel):
    profile: ProfileIn = ProfileIn()
    food_class: str
    grams: float
    confidence: float = 0.0
    source: str = "photo"


class AdviseIn(BaseModel):
    profile: ProfileIn = ProfileIn()
    food_class: str
    grams: float


class SummaryOut(BaseModel):
    date: str
    total: MacrosOut
    target_kcal: float
    summary_az: str
    plan_az: str


def _meal_to_out(meal: MealAnalysis) -> AnalysisOut:
    macros = MacrosOut(**meal.macros.__dict__) if meal.macros else None
    return AnalysisOut(ok=meal.ok, food_class=meal.food_class, az_name=meal.az_name,
                       confidence=meal.confidence, top5=meal.top5, grams=meal.grams,
                       portion_bucket=meal.portion_bucket,
                       portion_confidence=meal.portion_confidence, macros=macros,
                       tags=meal.tags, quality_reasons_az=meal.quality_reasons_az,
                       advice_az=meal.advice_az)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisOut)
async def analyze_endpoint(image: UploadFile = File(...)) -> AnalysisOut:
    """Multipart image -> full MealAnalysis."""
    data = await image.read()
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(400, "Şəkil oxuna bilmədi.")
    return _meal_to_out(analyze(arr))


@app.post("/parse", response_model=list[EntityOut])
def parse_endpoint(body: ParseIn) -> list[EntityOut]:
    """Free Azerbaijani text -> MealEntity list (NLP-only path)."""
    return [EntityOut(food=e.food, qty=e.qty, unit=e.unit, raw=e.raw,
                      in_db=e.in_db, matched_via=e.matched_via)
            for e in parse_meal(body.text)]


@app.post("/log")
def log_endpoint(body: LogIn) -> dict:
    """Persist one MealLog for the (auto-created) user."""
    if body.grams <= 0:
        raise HTTPException(400, "grams must be > 0")
    try:
        macros, _, _ = macros_for(body.food_class, body.grams)
    except KeyError:
        raise HTTPException(404, f"Unknown food_class: {body.food_class}")
    session = db.get_session()
    p = body.profile
    user = db.get_or_create_user(session, p.name, p.age, p.sex, p.height_cm,
                                 p.weight_kg, p.activity, p.goal)
    row = db.MealLog(user_id=user.id, food_class=body.food_class,
                     confidence=body.confidence, grams=body.grams,
                     kcal=macros.kcal, protein_g=macros.protein_g,
                     carb_g=macros.carb_g, fat_g=macros.fat_g,
                     sodium_mg=macros.sodium_mg, source=body.source)
    session.add(row)
    session.commit()
    return {"id": row.id, "user_id": user.id, "kcal": macros.kcal}


@app.post("/advise")
def advise_endpoint(body: AdviseIn) -> dict:
    """MealAnalysis-lite + profile -> Azerbaijani advice text."""
    try:
        macros, tags, az_name = macros_for(body.food_class, body.grams)
    except KeyError:
        raise HTTPException(404, f"Unknown food_class: {body.food_class}")
    meal = MealAnalysis(ok=True, food_class=body.food_class, az_name=az_name,
                        grams=body.grams, macros=macros, tags=tags)
    return {"advice_az": advise(meal, body.profile.to_profile())}


@app.get("/summary/{user_id}", response_model=SummaryOut)
def summary_endpoint(user_id: int, date: Optional[str] = None) -> SummaryOut:
    """Daily summary + next-day plan for a user."""
    day = dt.date.fromisoformat(date) if date else dt.date.today()
    session = db.get_session()
    user = session.get(db.User, user_id)
    if user is None:
        raise HTTPException(404, "User not found")
    logs = db.logs_for_date(session, user_id, day)
    profile = UserProfile(name=user.name, age=user.age, sex=user.sex,
                          height_cm=user.height_cm, weight_kg=user.weight_kg,
                          activity=user.activity, goal=user.goal,
                          daily_kcal_target=user.daily_kcal_target)
    s = daily_summary(logs, profile, date=day.isoformat())
    return SummaryOut(date=s.date, total=MacrosOut(**s.total.__dict__),
                      target_kcal=s.target_kcal, summary_az=s.summary_az,
                      plan_az=s.plan_az)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
