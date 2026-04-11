from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    @abstractmethod
    def generate_signals(self) -> dict[str, pd.Series]:
        raise NotImplementedError("Must implement generate_signals method")