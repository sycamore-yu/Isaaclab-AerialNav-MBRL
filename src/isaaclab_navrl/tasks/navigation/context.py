"""Explicit sensor and privileged observation sources, with no sensor side effects."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from torch import Tensor


@dataclass(frozen=True)
class ObservationContext:
    """Read-only namespaces over tensors produced by the current sensor tick.

    Namespace dictionaries are copied, not the tensors: producers own tensor
    lifetime and consumers must not mutate them. The policy map contains only
    the selected sensor features. Simulation truth and training metadata stay
    in the privileged map; neither map falls back to the other.
    """

    policy: Mapping[str, Tensor]
    privileged: Mapping[str, Tensor]

    def __post_init__(self):
        object.__setattr__(self, "policy", MappingProxyType(dict(self.policy)))
        object.__setattr__(self, "privileged", MappingProxyType(dict(self.privileged)))

    @classmethod
    def build(cls, *, policy: Mapping[str, Tensor], privileged: Mapping[str, Tensor]):
        return cls(policy=policy, privileged=privileged)
