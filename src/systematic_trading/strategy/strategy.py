from abc import ABC, abstractmethod
from typing import Any, Protocol
import pandas as pd


class Strategy(ABC):
    @abstractmethod
    def generate_signals(self) -> dict[str, pd.Series]:
        raise NotImplementedError("Must implement generate_signals method")


class OptimizableStrategyFactory(Protocol):
    def __call__(
        self,
        *args: Any,
        hyper_param_dict: dict | None = None,
        **kwargs: Any,
    ) -> Strategy: ...
