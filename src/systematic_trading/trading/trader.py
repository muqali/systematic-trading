from abc import ABC, abstractmethod
import pandas as pd


class Trader(ABC):

    @abstractmethod
    def generate_positions(self) -> dict[str, pd.Series]:

        raise NotImplementedError("Should implement generate_positions()!")

    @abstractmethod
    def generate_trades(self) -> dict[str, pd.DataFrame]:

        raise NotImplementedError("Should implement generate_trades()!")

    @abstractmethod
    def generate_net_pnl(self) -> dict[str, pd.Series]:

        raise NotImplementedError("Should implement generate_net_pnl()!")

    @abstractmethod
    def generate_gross_pnl(self) -> dict[str, pd.Series]:

        raise NotImplementedError("Should implement generate_gross_pnl()!")
