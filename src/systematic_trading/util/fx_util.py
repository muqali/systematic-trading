PIP_SIZE_DICT = {
    "EURUSD": 0.0001,
    "USDJPY": 0.01,
    "GBPUSD": 0.0001,
    "USDCHF": 0.0001,
    "USDNOK": 0.0001,
    "USDSEK": 0.0001,
    "AUDUSD": 0.0001,
    "NZDUSD": 0.0001,
    "USDCAD": 0.0001,
    "USDCNH": 0.0001,
    "USDSGD": 0.0001,
    "USDTHB": 0.01,
    "USDHKD": 0.0001,
    "USDTRY": 0.0001,
    "USDZAR": 0.0001,
    "USDILS": 0.0001,
    "USDPLN": 0.0001,
    "USDHUF": 0.01,
    "USDCZK": 0.001,
    "USDRON": 0.0001,
    "USDMXN": 0.0001,
    "EURNOK": 0.0001,
    "EURSEK": 0.0001,
    "EURPLN": 0.0001,
    "EURHUF": 0.01,
    "EURCZK": 0.001,
    "EURRON": 0.0001,
    "EURGBP": 0.0001,
    "EURJPY": 0.01,
    "EURCHF": 0.0001,
}

def get_pip_size(ticker: str) -> float:
    """
    Get the pip size for a given currency pair ticker.

    Args:
        ticker (str): The currency pair ticker (e.g., "EURUSD").

    Returns:
        float: The pip size for the given ticker.
    """
    return PIP_SIZE_DICT.get(ticker, None)