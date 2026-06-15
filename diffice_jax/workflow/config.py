from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkflowConfig:
    """Top-level config for a config-driven training workflow."""

    name: str
    workflow: str | None
    data: dict[str, Any]
    model: dict[str, Any]
    equation: dict[str, Any] = field(default_factory=dict)
    loss: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    legacy: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    base_dir: Path = Path(".")


def load_workflow_config(path: str | Path) -> WorkflowConfig:
    """Load a YAML workflow config, with JSON retained for legacy files."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML workflow configs require PyYAML.") from exc
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raise ValueError(f"Unsupported workflow config suffix: {path.suffix}")

    if raw is None:
        raw = {}
    return workflow_config_from_dict(raw, base_dir=path.parent)


def workflow_config_from_dict(raw: dict[str, Any], base_dir: str | Path = ".") -> WorkflowConfig:
    return WorkflowConfig(
        name=raw.get("name", "diffice_workflow"),
        workflow=raw.get("workflow"),
        runtime=dict(raw.get("runtime", {})),
        legacy=dict(raw.get("legacy", {})),
        data=dict(raw.get("data", {})),
        model=dict(raw.get("model", {})),
        equation=dict(raw.get("equation", {})),
        loss=dict(raw.get("loss", {})),
        training=dict(raw.get("training", {})),
        artifacts=dict(raw.get("artifacts", {})),
        base_dir=Path(base_dir),
    )
