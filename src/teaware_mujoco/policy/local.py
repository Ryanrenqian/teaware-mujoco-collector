from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import ActionChunk, PolicyObservation


class LocalModelPolicy:
    """Adapter for an in-process VLA/model callable."""

    def __init__(
        self,
        predict: Callable[[PolicyObservation], ActionChunk | dict[str, Any]],
        *,
        name: str,
        model_metadata: dict[str, Any] | None = None,
        reset: Callable[[int, PolicyObservation], None] | None = None,
        close: Callable[[], None] | None = None,
    ) -> None:
        self._predict = predict
        self._reset = reset
        self._close = close
        self._metadata = {"type": "local_vla", "name": name, **(model_metadata or {})}

    def reset(self, seed: int, initial_observation: PolicyObservation) -> None:
        if self._reset is not None:
            self._reset(seed, initial_observation)

    def act(self, observation: PolicyObservation) -> ActionChunk:
        result = self._predict(observation)
        return result if isinstance(result, ActionChunk) else ActionChunk.from_mapping(result)

    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def close(self) -> None:
        if self._close is not None:
            self._close()
