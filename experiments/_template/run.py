from __future__ import annotations

from pathlib import Path

from iat.io import read_yaml, snapshot_run, write_json, make_run_dir
from iat.seeding import set_seed


EXPERIMENT_DIR = Path(__file__).parent


def main() -> None:
    config = read_yaml(EXPERIMENT_DIR / "config.yaml")
    set_seed(config["seed"])

    run_dir = make_run_dir(EXPERIMENT_DIR)
    snapshot_run(run_dir, config)

    summary: dict[str, object] = {
        "experiment": config["experiment"]["id"],
        "status": "template_no_implementation",
    }
    write_json(run_dir / "summary.json", summary)

    print(f"run completado: {run_dir}")


if __name__ == "__main__":
    main()
