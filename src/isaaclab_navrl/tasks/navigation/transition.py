"""Owned pre-autoreset physical transitions and their one-shot consumer."""

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from torch import Tensor


def _snapshot(value):
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return MappingProxyType({key: _snapshot(item) for key, item in value.items()})
    raise TypeError(f"Transition fields must be tensors or tensor mappings, got {type(value).__name__}")


@dataclass(frozen=True)
class TransitionBatch(Mapping):
    """Detached old-episode data, independent of mutable simulator buffers.

    This map is read-only; its tensor values must also be treated as read-only.
    The returned next observation is deliberately separate: it may already be
    the initial observation of a new episode. Surrogate graphs belong to the
    trainer, not this physical snapshot.
    """

    _values: Mapping

    @classmethod
    def capture(cls, values: Mapping):
        required = {"position", "velocity", "episode_id", "executed_action", "terminated", "timeout", "step", "time_s"}
        missing = required.difference(values)
        if missing:
            raise ValueError(f"Incomplete pre-reset transition: missing {sorted(missing)}")
        return cls(_snapshot(values))

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self) -> Iterator:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    @property
    def done(self) -> Tensor:
        return self["terminated"] | self["timeout"]

    def with_fields(self, **values):
        """Add later reward bookkeeping without changing the published object."""
        return type(self)(
            MappingProxyType(dict(self._values) | {key: _snapshot(value) for key, value in values.items()})
        )


class TransitionPublisher:
    """Publish once before autoreset, without knowing the consumer's algorithm."""

    def __init__(self):
        self._consumer: Callable[[TransitionBatch], None] | None = None

    def set_consumer(self, consumer: Callable[[TransitionBatch], None]):
        if self._consumer is not None:
            raise RuntimeError("A pre-reset transition consumer is already pending")
        self._consumer = consumer

    def clear(self):
        self._consumer = None

    def publish(self, values: Mapping) -> TransitionBatch:
        transition = TransitionBatch.capture(values)
        consumer, self._consumer = self._consumer, None
        if consumer is not None:
            consumer(transition)
        return transition
