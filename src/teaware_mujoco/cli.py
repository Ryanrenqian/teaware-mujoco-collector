from __future__ import annotations

import argparse
import json
import shutil
from importlib.resources import files
from pathlib import Path

import uvicorn

from .collector import TeawareCollector
from .config import load_config
from .scene import write_scene_xml
from .schema import discover_episodes, validate_dataset
from .web import create_app


def default_config_path() -> Path:
    return Path(str(files("teaware_mujoco").joinpath("default_scene.yaml")))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="teaware-mj",
        description="MuJoCo teaware scene collection and browser.",
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Collect one or more episodes.")
    collect.add_argument("--output", type=Path, default=Path("data/teaware"))
    collect.add_argument("--episodes", type=int, default=1)
    collect.add_argument("--seed", type=int, default=0)

    serve = subparsers.add_parser("serve", help="Run the collection and inspection web UI.")
    serve.add_argument("--dataset", type=Path, default=Path("data/teaware"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)

    validate = subparsers.add_parser("validate", help="Validate every dataset episode.")
    validate.add_argument("--dataset", type=Path, default=Path("data/teaware"))

    scene = subparsers.add_parser("scene", help="Generate the effective MJCF scene XML.")
    scene.add_argument("--output", type=Path, default=Path("scene.generated.xml"))

    init_config = subparsers.add_parser("init-config", help="Copy the default YAML for editing.")
    init_config.add_argument("destination", type=Path, nargs="?", default=Path("tea_table.yaml"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "init-config":
        destination = args.destination.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        shutil.copyfile(default_config_path(), destination)
        print(destination)
        return 0

    config = load_config(args.config)
    if args.command == "scene":
        print(write_scene_xml(config, args.output))
        return 0
    if args.command == "collect":
        collector = TeawareCollector(config, args.output)
        try:
            paths = collector.collect(args.episodes, args.seed)
        finally:
            collector.close()
        for path in paths:
            print(path)
        return 0
    if args.command == "serve":
        app = create_app(config, args.dataset)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    if args.command == "validate":
        results = validate_dataset(args.dataset)
        if not results and not discover_episodes(args.dataset):
            print(f"no episodes found under {args.dataset}")
            return 1
        failures = {episode: errors for episode, errors in results.items() if errors}
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 1 if failures else 0
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
