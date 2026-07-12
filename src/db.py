"""SQLAlchemy models (User, MealLog) + Mifflin-St Jeor daily target."""

from __future__ import annotations

import datetime as dt
import logging
from typing import Optional

from sqlalchemy import (DateTime, Float, ForeignKey, Integer, String,
                        create_engine, select)
from sqlalchemy.orm import (DeclarativeBase, Mapped, Session, mapped_column,
                            relationship, sessionmaker)

from config import settings

log = logging.getLogger(__name__)

ACTIVITY_FACTORS = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55,
                    "active": 1.725, "very_active": 1.9}
GOAL_ADJUST = {"lose": -400.0, "maintain": 0.0, "gain": 400.0}


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    age: Mapped[int] = mapped_column(Integer)
    sex: Mapped[str] = mapped_column(String(1))  # "m" | "f"
    height_cm: Mapped[float] = mapped_column(Float)
    weight_kg: Mapped[float] = mapped_column(Float)
    activity: Mapped[str] = mapped_column(String(20), default="light")
    goal: Mapped[str] = mapped_column(String(10), default="maintain")
    daily_kcal_target: Mapped[float] = mapped_column(Float)

    logs: Mapped[list["MealLog"]] = relationship(back_populates="user")


class MealLog(Base):
    __tablename__ = "meal_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    ts: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.now)
    image_path: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    food_class: Mapped[str] = mapped_column(String(50))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    grams: Mapped[float] = mapped_column(Float)
    kcal: Mapped[float] = mapped_column(Float)
    protein_g: Mapped[float] = mapped_column(Float, default=0.0)
    carb_g: Mapped[float] = mapped_column(Float, default=0.0)
    fat_g: Mapped[float] = mapped_column(Float, default=0.0)
    sodium_mg: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(10), default="photo")  # photo|text

    user: Mapped[User] = relationship(back_populates="logs")


def daily_kcal_target(age: int, sex: str, height_cm: float, weight_kg: float,
                      activity: str = "light", goal: str = "maintain") -> float:
    """Mifflin-St Jeor BMR x activity factor, adjusted by goal."""
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if sex == "m" else -161)
    tdee = bmr * ACTIVITY_FACTORS.get(activity, 1.375)
    return round(tdee + GOAL_ADJUST.get(goal, 0.0))


_engine = None
_session_factory = None


def get_session() -> Session:
    """Create tables on first use and return a new Session."""
    global _engine, _session_factory
    if _engine is None:
        _engine = create_engine(settings.database_url, echo=False)
        Base.metadata.create_all(_engine)
        _session_factory = sessionmaker(bind=_engine)
    return _session_factory()


def get_or_create_user(session: Session, name: str, age: int, sex: str,
                       height_cm: float, weight_kg: float, activity: str,
                       goal: str) -> User:
    """Find a user by name or create one with a computed kcal target."""
    user = session.scalar(select(User).where(User.name == name))
    target = daily_kcal_target(age, sex, height_cm, weight_kg, activity, goal)
    if user is None:
        user = User(name=name, age=age, sex=sex, height_cm=height_cm,
                    weight_kg=weight_kg, activity=activity, goal=goal,
                    daily_kcal_target=target)
        session.add(user)
    else:
        user.age, user.sex, user.height_cm, user.weight_kg = age, sex, height_cm, weight_kg
        user.activity, user.goal, user.daily_kcal_target = activity, goal, target
    session.commit()
    session.refresh(user)
    return user


def logs_for_date(session: Session, user_id: int, date: dt.date) -> list[MealLog]:
    """All MealLogs of one user for one calendar day."""
    start = dt.datetime.combine(date, dt.time.min)
    end = dt.datetime.combine(date, dt.time.max)
    return list(session.scalars(
        select(MealLog).where(MealLog.user_id == user_id,
                              MealLog.ts >= start, MealLog.ts <= end)
        .order_by(MealLog.ts)))


if __name__ == "__main__":
    print("target (30yo m, 175cm, 75kg, light, maintain):",
          daily_kcal_target(30, "m", 175, 75))
    s = get_session()
    u = get_or_create_user(s, "Test", 30, "m", 175, 75, "light", "maintain")
    print(f"user id={u.id} target={u.daily_kcal_target}")
