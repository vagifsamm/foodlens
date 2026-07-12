"""Central configuration for FoodLens.

Everything path- or environment-dependent lives here. No module may
hardcode paths; import ``settings`` instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent

# 25-class Food101 subset (spec section 3.1).
CLASSES: list[str] = [
    "pizza", "hamburger", "french_fries", "caesar_salad", "sushi",
    "steak", "fried_rice", "spaghetti_bolognese", "pancakes", "omelette",
    "grilled_salmon", "chicken_curry", "donuts", "cheesecake", "ice_cream",
    "hot_dog", "dumplings", "falafel", "greek_salad", "lasagna",
    "ramen", "waffles", "tacos", "guacamole", "club_sandwich",
]

SEED = 42
IMG_SIZE = 224
NUM_CLASSES = len(CLASSES)


class Settings(BaseSettings):
    """Runtime settings, overridable via .env or environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_provider: Literal["template", "local", "anthropic"] = "template"
    anthropic_api_key: str = ""
    device: Literal["auto", "cpu", "cuda"] = "auto"
    num_workers: int = 0
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'foodlens.db'}"

    # Paths (derived, not env-driven)
    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def nutrition_db_path(self) -> Path:
        return self.data_dir / "nutrition_db.json"

    @property
    def guidelines_dir(self) -> Path:
        return self.data_dir / "guidelines"

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / "models"

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / "reports"

    def resolve_device(self) -> str:
        """Return the actual torch device string ('cuda' or 'cpu')."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


settings = Settings()

if __name__ == "__main__":
    print(f"root={PROJECT_ROOT}")
    print(f"classes={NUM_CLASSES}, provider={settings.llm_provider}, device={settings.resolve_device()}")
