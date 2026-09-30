"""Shared colors for speech-condition plots."""

from __future__ import annotations

CONDITION_ORDER = ("overt", "minimally_overt", "covert")
CONDITION_LABELS = {
    "overt": "overt",
    "minimally_overt": "min-overt",
    "covert": "covert",
}

_ALIASES = {
    "overt": "overt",
    "minimally_overt": "minimally_overt",
    "minimally overt": "minimally_overt",
    "min-overt": "minimally_overt",
    "min_overt": "minimally_overt",
    "minovert": "minimally_overt",
    "covert": "covert",
}

CONDITION_COLORS = {
    "overt": "#08306B",
    "minimally_overt": "#2171B5",
    "covert": "#6BAED6",
}


def normalize_condition(condition: str) -> str:
    text = str(condition).strip()
    key = text.lower().replace("-", "_")
    return _ALIASES.get(key, _ALIASES.get(text.lower(), text))


def condition_label(condition: str) -> str:
    key = normalize_condition(condition)
    return CONDITION_LABELS.get(key, str(condition))


def condition_color(condition: str):
    return CONDITION_COLORS[normalize_condition(condition)]


def condition_palette(conditions=None, *, labels: bool = False) -> dict:
    if conditions is None:
        conditions = CONDITION_ORDER
    return {
        condition_label(condition) if labels else condition: condition_color(condition)
        for condition in conditions
    }


def condition_colors(conditions=None) -> list:
    if conditions is None:
        conditions = CONDITION_ORDER
    return [condition_color(condition) for condition in conditions]
