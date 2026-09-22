from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

MASS_SCALE = 0.1095501188853104
MAX_MASS_KG = 0.1


def teaware_asset_dir() -> Path:
    return Path(str(files("teaware_mujoco").joinpath("assets", "teaware")))


@lru_cache(maxsize=1)
def _catalog() -> dict[str, dict[str, Any]]:
    catalog_path = teaware_asset_dir() / "catalog.json"
    with catalog_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {entry["asset_id"]: entry for entry in payload["assets"]}


def available_asset_ids() -> tuple[str, ...]:
    return tuple(sorted(_catalog()))


def get_teaware_asset(asset_id: str) -> dict[str, Any]:
    try:
        entry = _catalog()[asset_id]
    except KeyError as exc:
        raise KeyError(f"unknown teaware asset_id: {asset_id}") from exc

    root = teaware_asset_dir()
    required_paths = [entry["visual"], *entry["collisions"]]
    missing = [relative for relative in required_paths if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"teaware asset {asset_id!r} is incomplete: {missing[0]}")
    return entry


def effective_mass_kg(asset: dict[str, Any]) -> float:
    return min(float(asset["mass_kg"]) * MASS_SCALE, MAX_MASS_KG)
