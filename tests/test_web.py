from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from teaware_mujoco.web import create_app


def test_web_lists_and_renders_episode(collected_dataset: Path, tiny_config: dict) -> None:
    app = create_app(tiny_config, collected_dataset)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        status = client.get("/api/status").json()
        assert status["episode_count"] == 1
        assert status["cameras"] == ["front_left"]
        assert status["policy"]["type"] == "scripted_motion"

        episodes = client.get("/api/episodes").json()
        assert episodes[0]["episode_id"] == "episode_000000"
        assert episodes[0]["valid"] is True
        assert episodes[0]["grasp_outcome"] is None

        image = client.get(
            "/api/episodes/episode_000000/image",
            params={
                "frame": "000000",
                "camera": "front_left",
                "modality": "segmentation",
            },
        )
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/png"


def test_web_randomizes_and_collects(tmp_path: Path, tiny_config: dict) -> None:
    app = create_app(tiny_config, tmp_path / "dataset")
    with TestClient(app) as client:
        assert client.post("/api/randomize", json={"seed": 9}).json()["seed"] == 9
        live = client.get("/api/live/front_left/rgb")
        assert live.status_code == 200
        assert live.headers["content-type"] == "image/jpeg"
        result = client.post("/api/collect", json={"episodes": 1, "seed": 30}).json()
        assert result["episodes"] == ["episode_000000"]
