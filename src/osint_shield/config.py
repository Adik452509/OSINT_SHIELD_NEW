"""YAML configuration loading with single-level inheritance.

A config may declare ``extends: base.yaml`` and override any subset of the
parent's keys. Merging is recursive for mappings and replacing for scalars
and lists, so overriding ``training.batch_size`` leaves the rest of
``training`` intact while overriding ``seeds`` replaces the whole list.

    >>> cfg = load_config("mmbert_small.yaml")
    >>> cfg["model"]["name"]
    'jhu-clsp/mmBERT-small'
    >>> cfg["training"]["epochs"]      # inherited from base.yaml
    5
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIGS, resolve

MAX_EXTENDS_DEPTH = 5


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``, returning a new dict.

    Nested mappings are merged key by key; every other type is replaced
    outright. Neither input is mutated.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping at the top level: {path}")
    return data


def load_config(name: str | Path, _depth: int = 0) -> dict[str, Any]:
    """Load a config by filename (resolved against ``configs/``) or by path.

    Follows a single ``extends`` key per file, recursively, up to
    :data:`MAX_EXTENDS_DEPTH` levels to catch cycles.
    """
    if _depth > MAX_EXTENDS_DEPTH:
        raise ValueError(f"config 'extends' chain too deep - cycle in {name}?")

    path = Path(name)
    if not path.is_absolute() and not path.exists():
        path = CONFIGS / path
    cfg = _read_yaml(path)

    parent_name = cfg.pop("extends", None)
    if parent_name is None:
        return cfg

    parent = load_config(parent_name, _depth=_depth + 1)
    return deep_merge(parent, cfg)


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides, returning a new config.

    Values are parsed as YAML, so ``false``, ``1024`` and ``2.0e-5`` arrive as
    a bool, an int and a float. Every key must already exist: a typo such as
    ``training.early_stoping=false`` raises instead of silently running the
    reference configuration under an ablation's name.

        >>> apply_overrides(cfg, ["tokenizer.max_length=1024"])

    Raises:
        ValueError: an override is not of the form ``key=value``.
        KeyError: the dotted path does not exist in ``cfg``.
    """
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        path, raw = item.split("=", 1)
        keys = path.strip().split(".")
        node: Any = out
        for depth, key in enumerate(keys):
            if not isinstance(node, dict) or key not in node:
                where = ".".join(keys[:depth]) or "<top level>"
                raise KeyError(f"unknown config key {path!r}: no {key!r} under {where}")
            if depth == len(keys) - 1:
                node[key] = _parse_override_value(raw)
            else:
                node = node[key]
    return out


_SCI_NOTATION = re.compile(r"^[+-]?\d+(\.\d*)?[eE][+-]?\d+$")


def _parse_override_value(raw: str) -> Any:
    """YAML-parse a value, also accepting ``5e-5`` as a float.

    YAML 1.1 only reads scientific notation with a decimal point (``5.0e-5``);
    ``5e-5`` would otherwise arrive as the string ``"5e-5"``.
    """
    raw = raw.strip()
    if raw == "":
        return ""      # `key=` means an empty string; YAML would read it as null
    if _SCI_NOTATION.match(raw):
        return float(raw)
    return yaml.safe_load(raw)


def load_taxonomy() -> dict[str, Any]:
    """Load the label space from ``data/resources/taxonomy.yaml``."""
    from .paths import TAXONOMY_YAML

    return _read_yaml(TAXONOMY_YAML)


def narrative_label_maps() -> tuple[dict[str, int], dict[int, str]]:
    """Return ``(name -> id, id -> name)`` for the trained narrative classes.

    Built from ``taxonomy.yaml`` rather than the corpus's own
    ``narrative_label`` column, whose ids are non-contiguous (3 is unused).
    Contiguous ids keep a dead logit out of the classification head.
    """
    classes = load_taxonomy()["narrative"]["classes"]
    name_to_id = {c["name"]: int(c["id"]) for c in classes}
    id_to_name = {v: k for k, v in name_to_id.items()}
    if sorted(id_to_name) != list(range(len(classes))):
        raise ValueError(f"narrative ids must be contiguous from 0: {sorted(id_to_name)}")
    return name_to_id, id_to_name


def severity_label_maps() -> tuple[dict[str, int], dict[int, str]]:
    """Return ``(name -> id, id -> name)`` for severity."""
    classes = load_taxonomy()["severity"]["classes"]
    name_to_id = {c["name"]: int(c["id"]) for c in classes}
    return name_to_id, {v: k for k, v in name_to_id.items()}


def collapse_map() -> dict[str, str]:
    """Return the 5-class -> 3-class narrative collapse map."""
    return dict(load_taxonomy()["narrative"]["collapse_3"])


def config_path(cfg: dict[str, Any], *keys: str) -> Path:
    """Read a dotted config value and resolve it as a repo-relative path."""
    node: Any = cfg
    for k in keys:
        node = node[k]
    return resolve(node)
