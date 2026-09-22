from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


@contextmanager
def _tro_import_context(root: Path):
    previous_cwd = Path.cwd()
    root_text = str(root)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        if inserted:
            sys.path.remove(root_text)


def _load_class(path: str) -> type:
    module_name, class_name = path.split(":", 1) if ":" in path else path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _configured_path(value: str | Path, label: str) -> Path:
    raw = str(value)
    expanded = os.path.expandvars(raw)
    if "$" in expanded:
        raise ValueError(f"unresolved environment variable in TRO {label}: {raw}")
    return Path(expanded).expanduser().resolve()


class LocalTRORuntime:
    """Minimal standalone TRO loader; it imports only the configured TRO checkout."""

    def __init__(
        self,
        *,
        root: str | Path,
        config: str | Path,
        checkpoint: str | Path,
        hand_type: str = "xhand",
        device: str | None = None,
        num_candidates: int = 8,
        inference_steps: int | None = None,
        noise_lambda: float | None = None,
        root_link_name: str | None = None,
    ) -> None:
        try:
            import torch
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise RuntimeError(
                "local TRO requires torch and omegaconf; install with `uv sync --extra tro`"
            ) from exc

        self.torch = torch
        self.OmegaConf = OmegaConf
        self.root = _configured_path(root, "root")
        self.config_path = _configured_path(config, "config")
        self.checkpoint = _configured_path(checkpoint, "checkpoint")
        self.hand_type = str(hand_type)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_candidates = int(num_candidates)
        self.root_link_name = root_link_name or {
            "xhand": "right_hand_link",
            "xhand_left": "left_hand_link",
        }.get(self.hand_type, "right_hand_link")
        if not self.root.is_dir():
            raise FileNotFoundError(f"TRO root does not exist: {self.root}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"TRO config does not exist: {self.config_path}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"TRO checkpoint does not exist: {self.checkpoint}")
        if self.num_candidates <= 0:
            raise ValueError("TRO num_candidates must be positive")

        with _tro_import_context(self.root):
            self.config = OmegaConf.load(self.config_path)
            self.config.test.ckpt = str(self.checkpoint)
            self.config.test.embodiment = self.hand_type
            if "dataset" in self.config:
                self.config.dataset.hand_type = self.hand_type
            self.config.model.mode = "test"
            if inference_steps is not None:
                self.config.model.inference_config.inference_step = int(inference_steps)
            if noise_lambda is not None:
                self.config.model.diffusion_config["lambda"] = float(noise_lambda)

            from model.vqvae_encoder import VQVAEEncoder

            class_path = (
                OmegaConf.select(self.config, "test.robot_graph_class")
                or OmegaConf.select(self.config, "model.robot_graph_class")
                or "model.troe_graph_v5:RobotGraph"
            )
            graph_class = _load_class(str(class_path))
            self.encoder = VQVAEEncoder(**self.config.vqvae).to(self.device).eval()
            model_kwargs = OmegaConf.create(OmegaConf.to_container(self.config.model, resolve=True))
            model_kwargs.pop("robot_graph_class", None)
            self.model = graph_class(**model_kwargs).to(self.device).eval()
            checkpoint_payload = torch.load(self.checkpoint, map_location="cpu")
            self.model.load_state_dict(checkpoint_payload["model_state"], strict=False)
            del checkpoint_payload

    def infer(
        self, object_points: np.ndarray, environment_points: np.ndarray
    ) -> list[dict[str, Any]]:
        torch = self.torch
        object_array = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
        environment_array = np.asarray(environment_points, dtype=np.float32).reshape(-1, 3)
        object_batch = np.ascontiguousarray(
            np.broadcast_to(object_array[None], (self.num_candidates, *object_array.shape))
        )
        environment_batch = np.ascontiguousarray(
            np.broadcast_to(
                environment_array[None],
                (self.num_candidates, *environment_array.shape),
            )
        )
        embodiments = list(self.config.model.embodiment)
        if self.hand_type not in embodiments:
            raise ValueError(
                f"TRO hand type {self.hand_type!r} is not in config embodiments {embodiments}"
            )
        robot_id = embodiments.index(self.hand_type)
        batch = {
            "robot_id": torch.full(
                (self.num_candidates,), robot_id, dtype=torch.long, device=self.device
            ),
            "object_pc": torch.from_numpy(object_batch).to(self.device),
            "env_pc": torch.from_numpy(environment_batch).to(self.device),
        }
        with _tro_import_context(self.root), torch.no_grad():
            batch = self.encoder(batch)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                stages = self.model.inference(batch)
        predicted = stages[-1]
        if self.root_link_name not in predicted:
            raise KeyError(
                f"TRO output has no root link {self.root_link_name!r}; "
                f"available links: {sorted(predicted)}"
            )
        arrays = {
            name: value.detach().float().cpu().numpy()
            for name, value in predicted.items()
            if getattr(value, "ndim", 0) >= 3 and value.shape[-2:] == (4, 4)
        }
        candidates: list[dict[str, Any]] = []
        for index in range(self.num_candidates):
            candidates.append(
                {
                    "rank": index,
                    "root_link_name": self.root_link_name,
                    "root_pose_base": arrays[self.root_link_name][index],
                    "pred_links": {name: value[index] for name, value in arrays.items()},
                }
            )
        return candidates

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "local",
            "root": str(self.root),
            "config": str(self.config_path),
            "checkpoint": str(self.checkpoint),
            "hand_type": self.hand_type,
            "device": str(self.device),
            "num_candidates": self.num_candidates,
        }

    def close(self) -> None:
        self.encoder = None
        self.model = None
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()
