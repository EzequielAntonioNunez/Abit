from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def make_run_dir(experiment_dir: Path) -> Path:
    """
    Crea un directorio con timestamp ISO dentro de results/ del experimento.
    Devuelve la ruta creada.
    """
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = experiment_dir / "results" / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data: dict[str, Any] = yaml.safe_load(f)
        return data


def write_json(path: Path, data: object) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_git_sha(path: Path) -> None:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"
    path.write_text(sha + "\n")


def write_env_freeze(path: Path) -> None:
    try:
        out = subprocess.check_output(
            ["uv", "pip", "freeze"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            out = subprocess.check_output(
                ["pip", "freeze"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            out = "unknown\n"
    path.write_text(out)


def snapshot_run(run_dir: Path, config: dict[str, Any]) -> None:
    """
    Escribe en run_dir: config.snapshot.yaml, git_sha.txt, env.txt.
    Llamar al inicio de cada run, antes de ejecutar nada pesado.
    """
    write_yaml(run_dir / "config.snapshot.yaml", config)
    write_git_sha(run_dir / "git_sha.txt")
    write_env_freeze(run_dir / "env.txt")
