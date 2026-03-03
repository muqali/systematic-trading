from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    def generate_signals(self):
        raise NotImplementedError("Must implement generate_signals method")