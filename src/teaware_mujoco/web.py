from __future__ import annotations

import io
import queue
import secrets
import threading
from concurrent.futures import Future
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import __version__
from .collector import TeawareCollector, _segmentation_preview
from .schema import discover_episodes, load_manifest, validate_episode


class RandomizeRequest(BaseModel):
    seed: int | None = None


class CollectRequest(BaseModel):
    episodes: int = Field(default=1, ge=1, le=100)
    seed: int | None = None


class SimulationWorker:
    """Own the MuJoCo model and OpenGL context on one dedicated thread."""

    def __init__(self, config: dict[str, Any], dataset_root: str | Path):
        self._config = config
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self._queue: queue.Queue[tuple[str, tuple[Any, ...], Future[Any]] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="teaware-mujoco", daemon=True)
        self._ready: Future[None] = Future()
        self._thread.start()
        self._ready.result()

    def _run(self) -> None:
        collector: TeawareCollector | None = None
        try:
            collector = TeawareCollector(self._config, self.dataset_root)
            self._ready.set_result(None)
            while True:
                task = self._queue.get()
                if task is None:
                    break
                method, args, future = task
                if future.cancelled():
                    continue
                try:
                    future.set_result(getattr(collector, method)(*args))
                except Exception as exc:  # noqa: BLE001 - return simulation failures to caller
                    future.set_exception(exc)
        except Exception as exc:  # noqa: BLE001 - propagate worker initialization failures
            if not self._ready.done():
                self._ready.set_exception(exc)
        finally:
            if collector is not None:
                collector.close()

    def call(self, method: str, *args: Any) -> Any:
        if not self._thread.is_alive():
            raise RuntimeError("MuJoCo simulation worker is not running")
        future: Future[Any] = Future()
        self._queue.put((method, args, future))
        return future.result()

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError("MuJoCo simulation worker did not stop")


def _image_bytes(array: np.ndarray, image_format: str) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=image_format)
    return buffer.getvalue()


def _depth_preview(depth: np.ndarray) -> np.ndarray:
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size == 0:
        return np.zeros(depth.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [2.0, 98.0])
    high = max(float(high), float(low) + 1e-6)
    normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
    return np.asarray((1.0 - normalized) * 255.0, dtype=np.uint8)


def create_app(config: dict[str, Any], dataset_root: str | Path) -> FastAPI:
    dataset_path = Path(dataset_root).expanduser().resolve()
    static_path = Path(str(files("teaware_mujoco").joinpath("static")))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker = SimulationWorker(config, dataset_path)
        app.state.simulation_worker = worker
        try:
            yield
        finally:
            worker.close()

    app = FastAPI(title="Teaware MuJoCo Collector", version=__version__, lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=static_path), name="static")

    def simulation() -> SimulationWorker:
        return app.state.simulation_worker

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_path / "index.html")

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        simulation_status = simulation().call("status")
        episodes = discover_episodes(dataset_path)
        return {**simulation_status, "episode_count": len(episodes)}

    @app.post("/api/randomize")
    def randomize(request: RandomizeRequest) -> dict[str, Any]:
        seed = request.seed if request.seed is not None else secrets.randbelow(2**31)
        return simulation().call("randomize", seed)

    @app.post("/api/collect")
    def collect(request: CollectRequest) -> dict[str, Any]:
        seed = request.seed if request.seed is not None else secrets.randbelow(2**31)
        paths = simulation().call("collect", request.episodes, seed)
        return {"seed": seed, "episodes": [path.name for path in paths]}

    @app.get("/api/live/{camera}/{modality}")
    def live_image(
        camera: str,
        modality: Literal["rgb", "depth", "segmentation"],
    ) -> Response:
        status = simulation().call("status")
        if camera not in status["cameras"]:
            raise HTTPException(status_code=404, detail="unknown camera")
        images = simulation().call("latest_frames")[camera]
        if modality == "rgb":
            payload = _image_bytes(images["rgb"], "JPEG")
            return Response(payload, media_type="image/jpeg")
        if modality == "depth":
            payload = _image_bytes(_depth_preview(images["depth"]), "PNG")
        else:
            payload = _image_bytes(_segmentation_preview(images["instance"]), "PNG")
        return Response(payload, media_type="image/png")

    @app.get("/api/episodes")
    def episodes() -> list[dict[str, Any]]:
        output = []
        for episode_dir in reversed(discover_episodes(dataset_path)):
            manifest = load_manifest(episode_dir)
            output.append(
                {
                    "episode_id": manifest["episode_id"],
                    "created_at": manifest["created_at"],
                    "seed": manifest["seed"],
                    "frame_count": len(manifest["frames"]),
                    "valid": not validate_episode(episode_dir),
                    "grasp_outcome": manifest.get("grasp_outcome"),
                }
            )
        return output

    def resolve_episode(episode_id: str) -> tuple[Path, dict[str, Any]]:
        if not episode_id.startswith("episode_") or not episode_id[8:].isdigit():
            raise HTTPException(status_code=404, detail="unknown episode")
        episode_dir = dataset_path / "episodes" / episode_id
        if not episode_dir.is_dir():
            raise HTTPException(status_code=404, detail="unknown episode")
        return episode_dir, load_manifest(episode_dir)

    @app.get("/api/episodes/{episode_id}")
    def episode_detail(episode_id: str) -> dict[str, Any]:
        episode_dir, manifest = resolve_episode(episode_id)
        return {"manifest": manifest, "validation_errors": validate_episode(episode_dir)}

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/episodes/{episode_id}/image")
    def episode_image(
        episode_id: str,
        frame: str = Query(pattern=r"^\d{6}$"),
        camera: str = Query(min_length=1),
        modality: Literal["rgb", "depth", "segmentation"] = "rgb",
    ) -> FileResponse:
        episode_dir, manifest = resolve_episode(episode_id)
        if camera not in manifest["cameras"]:
            raise HTTPException(status_code=404, detail="unknown camera")
        if frame not in {item["frame_id"] for item in manifest["frames"]}:
            raise HTTPException(status_code=404, detail="unknown frame")
        suffix = {
            "rgb": "_rgb.jpg",
            "depth": "_depth_preview.png",
            "segmentation": "_segmentation_preview.png",
        }[modality]
        path = episode_dir / "frames" / frame / f"{camera}{suffix}"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="image not found")
        return FileResponse(path)

    return app
