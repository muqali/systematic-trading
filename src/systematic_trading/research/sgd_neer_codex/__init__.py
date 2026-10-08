"""SGD NEER research, companion notebooks and saved backtest results."""

import sys

from . import sgd_neer_backtest

# Existing local pickle caches contain BasketData/Policy instances recorded
# before this package was created. Keep them readable without rewriting the
# data or invalidating the historical input hashes in the saved results.
sys.modules.setdefault("research.sgd_neer_backtest", sgd_neer_backtest)
