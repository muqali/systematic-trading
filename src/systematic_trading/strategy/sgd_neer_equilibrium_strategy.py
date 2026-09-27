"""Index EWMA reversion with an aligned subset of synthetic SGD crosses.

This is the configurable strategy counterpart of the constituent-selection
research. No return regression or future-return target is fitted. All durations
count scheduled market time (Sunday 18:00 through Friday 17:00 New York), with
missing market bars retained. A day means 24 market hours.
"""
import numpy as np
import pandas as pd

from research.sgd_neer import DEFAULT_WEIGHTS, normalised_weights
from strategy.strategy import Strategy


class SGDNEEREquilibriumStrategy(Strategy):
    """Fade the index; freeze aligned constituent exposures until its exit.

    Constructor follows SGDNEERStrategy: (traded_instruments, fx_price_dict,
    hyper_param_dict=None). Quotes require timezone-aware, sorted, unique
    timestamps and bid/mid/ask columns. Supply USDSGD and USD-quoted constituents.

    Hyperparameters (durations accept Timedelta or strings):
      freq: inferred from quotes by default; explicit '1min' or '15min' recommended.
      equilibrium_half_life: '24h'; past EWMA reference, lagged one bar.
      equilibrium_min_history: max('20D', 2 * half-life) by default.
      zscore_half_life: '63D'; dispersion of past index deviations.
      zscore_min_history: '40D'; available observations required for dispersion.
      zscore_entry_threshold: 1.5; absolute entry threshold, inclusive.
      zscore_exit_threshold: 0.25; exit near/crossing the reference mean.
      max_holding_period: 2 * equilibrium_half_life by default (None);
                          'unlimited' disables time exits, retaining z-score exits.
      reversion_fraction: 0.5; heuristic fraction of displacement used as edge.
      margin_multiple: 1.5; edge must exceed this many full index spreads.
      volatility_horizon: '24h'; return window for risk and covariance estimates.
      volatility_half_life: '20D'; EWM risk-estimator half-life.
      volatility_min_history: '40D'; available return observations required.
      vol_target: 0.05; annual index volatility target, using sqrt(252) for 24h
                  returns, scaled by sqrt(24h / volatility_horizon) otherwise.
      max_units: 3.; cap aggregate SGD-cross units, including after risk matching.
      constituent_selection: 'aligned' (same-sign contributors), or 'full'.
      sizing_mode: 'risk_matched', 'retained', or 'normalized'.
      weights: quoted-pair weight mapping, DEFAULT_WEIGHTS by default; normalized
              over traded_instruments. Extra priced instruments are ignored.

    An entry still requires the *full index* signal and cost hurdle. Components
    are chosen once, at that entry. Index exits/48h default timeouts govern the
    entire selected portfolio. Positions survive overnight/weekend closures.

    generate_signals() returns unshifted pair targets on each input quote index,
    carrying positions through bars without a common quote. For synchronized
    backtests, pass backtest_prices to AggressiveTrader and choose execution
    delay there. Final liquidation also belongs to the backtest, not the causal
    strategy. No financing is calculated here.

    Diagnostics populated by generate_signals():
      features: index level, reference, deviation, zscore, spread and volatility.
      component_deviations: each synthetic SGD cross's deviation, on market grid.
      index_events: sparse full-index entry/exit requests (before selection).
      pair_targets: sparse selected-pair requests, including missing-quote exits.
      common_signals: requested positions filled onto common valid quote times.
      entry_details: selected flags, contributions, exposures and estimated risk.
      backtest_prices: common valid quotes, with gaps retained and no filling.
    """

    DEFAULT_HYPER_PARAMS = {
        "freq": None,
        "equilibrium_half_life": "24h",
        "equilibrium_min_history": None,
        "zscore_half_life": "63D",
        "zscore_min_history": "40D",
        "zscore_entry_threshold": 1.5,
        "zscore_exit_threshold": 0.25,
        "max_holding_period": None,
        "reversion_fraction": 0.5,
        "margin_multiple": 1.5,
        "volatility_horizon": "24h",
        "volatility_half_life": "20D",
        "volatility_min_history": "40D",
        "vol_target": 0.05,
        "max_units": 3.,
        "constituent_selection": "aligned",
        "sizing_mode": "risk_matched",
        "weights": None,
    }

    def __init__(self, traded_instruments, fx_price_dict, hyper_param_dict=None):
        supplied = dict(hyper_param_dict or {})
        unknown = supplied.keys()-self.DEFAULT_HYPER_PARAMS.keys()
        if unknown:
            raise ValueError(f"Unknown equilibrium hyperparameters: {sorted(unknown)}")
        params = {**self.DEFAULT_HYPER_PARAMS, **supplied}
        self.traded_instruments = tuple(traded_instruments)
        self.fx_price_dict = fx_price_dict
        if not self.traded_instruments or len(set(self.traded_instruments)) != len(self.traded_instruments):
            raise ValueError("traded_instruments must be nonempty and unique.")
        if "USDSGD" not in self.traded_instruments:
            raise ValueError("USDSGD is required to construct and hedge SGD crosses.")
        currencies = []
        for pair in self.traded_instruments:
            if len(pair) != 6 or not (pair.startswith("USD") or pair.endswith("USD")):
                raise ValueError(f"Expected a six-letter USD-quoted FX pair: {pair}")
            ccy = "USD" if pair == "USDSGD" else pair[3:] if pair.startswith("USD") else pair[:3]
            if ccy in currencies or ccy == "SGD":
                raise ValueError(f"Duplicate or invalid SGD-cross constituent: {pair}")
            currencies.append(ccy)
            if pair not in fx_price_dict:
                raise ValueError(f"Missing quotes for {pair}.")
            quotes = fx_price_dict[pair]
            if not {"bid", "mid", "ask"}.issubset(quotes.columns):
                raise ValueError(f"{pair}: bid, mid and ask columns are required.")
            idx = quotes.index
            if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
                raise ValueError(f"{pair}: timezone-aware DatetimeIndex required.")
            if len(idx) < 2 or not idx.is_monotonic_increasing or idx.has_duplicates:
                raise ValueError(f"{pair}: at least two sorted, unique quotes required.")
        if params["freq"] is None:
            params["freq"] = min(fx_price_dict[p].index.to_series().diff().dropna().min()
                                 for p in self.traded_instruments)
        self.freq = pd.Timedelta(params["freq"])
        if (pd.isna(self.freq) or self.freq < pd.Timedelta(minutes=1) or
                pd.Timedelta(hours=1) % self.freq != pd.Timedelta(0)):
            raise ValueError("freq must be a whole-minute divisor of one hour.")
        if self.freq % pd.Timedelta(minutes=1):
            raise ValueError("freq must contain a whole number of minutes.")
        for pair in self.traded_instruments:
            ticks = fx_price_dict[pair].index.tz_convert("UTC").as_unit("ns").asi8
            if (ticks % self.freq.value != 0).any():
                raise ValueError(f"{pair}: quote timestamps are not aligned to freq={self.freq}.")
        duration_keys = ["equilibrium_half_life", "zscore_half_life", "zscore_min_history",
                         "volatility_horizon", "volatility_half_life", "volatility_min_history"]
        for key in duration_keys:
            setattr(self, key, pd.Timedelta(params[key]))
        self.equilibrium_min_history = pd.Timedelta(params["equilibrium_min_history"] if
            params["equilibrium_min_history"] is not None else max(pd.Timedelta(days=20), 2*self.equilibrium_half_life))
        holding = params["max_holding_period"]
        self.max_holding_period = ("unlimited" if holding == "unlimited" else
            pd.Timedelta(holding if holding is not None else 2*self.equilibrium_half_life))
        for key in duration_keys+["equilibrium_min_history"]:
            self._bars(getattr(self, key), key)
        if self.max_holding_period != "unlimited":
            self._bars(self.max_holding_period, "max_holding_period")
        for key in ["zscore_min_history", "volatility_min_history"]:
            if self._bars(getattr(self, key), key) < 2:
                raise ValueError(f"{key} must span at least two bars.")
        for key in ["zscore_entry_threshold", "zscore_exit_threshold", "reversion_fraction",
                    "margin_multiple", "vol_target", "max_units"]:
            value = float(params[key])
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative.")
            setattr(self, key, value)
        if self.zscore_entry_threshold <= self.zscore_exit_threshold:
            raise ValueError("zscore_entry_threshold must exceed zscore_exit_threshold.")
        if min(self.reversion_fraction, self.vol_target, self.max_units) <= 0:
            raise ValueError("reversion_fraction, vol_target and max_units must be positive.")
        self.constituent_selection = params["constituent_selection"]
        self.sizing_mode = params["sizing_mode"]
        if self.constituent_selection not in ("aligned", "full"):
            raise ValueError("constituent_selection must be 'aligned' or 'full'.")
        if self.sizing_mode not in ("risk_matched", "retained", "normalized"):
            raise ValueError("sizing_mode must be 'risk_matched', 'retained' or 'normalized'.")
        self.weights = normalised_weights(list(self.traded_instruments),
                                           DEFAULT_WEIGHTS if params["weights"] is None else dict(params["weights"]))
        self.component_weights = pd.Series(list(self.weights.values()), index=currencies, name="weight")
        self.cross_to_pair = pd.DataFrame(0., index=currencies, columns=self.traded_instruments)
        self.cross_to_pair["USDSGD"] = -1.
        for ccy, pair in zip(currencies, self.traded_instruments):
            if pair != "USDSGD":
                self.cross_to_pair.loc[ccy, pair] = 1. if pair.startswith("USD") else -1.
        self.hyper_param_dict = {key: getattr(self, key) for key in params}

    def _bars(self, duration, name="duration"):
        duration = pd.Timedelta(duration)
        if pd.isna(duration) or duration <= pd.Timedelta(0) or duration % self.freq:
            raise ValueError(f"{name} must be positive and an exact multiple of freq.")
        return int(duration/self.freq)

    def _prepare(self):
        pairs = self.traded_instruments
        mids = pd.DataFrame({p: self.fx_price_dict[p]["mid"] for p in pairs})
        bids = pd.DataFrame({p: self.fx_price_dict[p]["bid"] for p in pairs})
        asks = pd.DataFrame({p: self.fx_price_dict[p]["ask"] for p in pairs})
        valid = ((bids > 0) & (asks >= bids) & (mids >= bids) & (mids <= asks)).all(axis=1)
        valid &= np.isfinite(mids).all(axis=1) & np.isfinite(bids).all(axis=1) & np.isfinite(asks).all(axis=1)
        local = mids.index.tz_convert("America/New_York")
        weekday, hour = local.dayofweek, local.hour
        valid &= (weekday < 4) | ((weekday == 4) & (hour < 17)) | ((weekday == 6) & (hour >= 18))
        mids, bids, asks = (frame.loc[valid] for frame in (mids, bids, asks))
        if len(mids) < 2:
            raise ValueError("At least two common valid quotes during the market week are required.")
        self.backtest_prices = {p: self.fx_price_dict[p].reindex(mids.index).copy() for p in pairs}
        spreads = (asks-bids)/mids
        local = mids.index.tz_convert("America/New_York")
        grid = pd.date_range(local.min(), local.max(), freq=self.freq)
        weekday, hour = grid.dayofweek, grid.hour
        grid = grid[(weekday < 4) | ((weekday == 4) & (hour < 17)) | ((weekday == 6) & (hour >= 18))]
        log_prices = np.log(mids)
        levels = log_prices.dot(self.cross_to_pair.T).reindex(grid)
        loading = self.component_weights.dot(self.cross_to_pair)
        loading["USDSGD"] = -1.
        index_level = log_prices.mul(loading).sum(axis=1).reindex(grid)
        spread = spreads.mul(loading.abs()).sum(axis=1).reindex(grid)
        return mids.index, levels, index_level, spreads, spread

    def _index_requests(self):
        frame = self.features
        annualization = np.sqrt(252*pd.Timedelta(days=1)/self.volatility_horizon)
        risk = (self.vol_target/(frame.volatility*annualization)).clip(upper=self.max_units)
        z, edge, units = (x.to_numpy() for x in (frame.zscore, frame.edge, risk))
        valid = frame.level.notna().to_numpy() & frame.spread.notna().to_numpy()
        usable = valid & np.isfinite(z) & np.isfinite(edge) & np.isfinite(units)
        entry = usable & (np.abs(z) >= self.zscore_entry_threshold) & (
            np.abs(edge) > self.margin_multiple*frame.spread.to_numpy())
        limit = np.inf if self.max_holding_period == "unlimited" else self._bars(self.max_holding_period)
        times, values = [0], [0.]
        q, entered, pending_close = 0., 0, False
        for i in range(len(frame)):
            previous = q
            if q and i-entered >= limit:
                q, pending_close = 0., not valid[i]
            elif pending_close:
                if valid[i]:
                    pending_close = False
            elif q:
                if usable[i] and q*z[i] >= -abs(q)*self.zscore_exit_threshold:
                    q = 0.
            elif entry[i]:
                q, entered = np.sign(edge[i])*units[i], i
            if q != previous:
                times.append(i)
                values.append(q)
        return pd.Series(values, index=frame.index[times], name="index_units").groupby(level=0).last()

    def _entry_covariance(self, moves, entries):
        covariance = np.empty((len(entries), len(moves.columns), len(moves.columns)))
        half_life = self._bars(self.volatility_half_life)
        minimum = self._bars(self.volatility_min_history)
        for i, ccy in enumerate(moves.columns):
            estimate = moves[ccy].ewm(halflife=half_life, min_periods=minimum)
            for j in range(i+1):
                values = estimate.cov(moves.iloc[:, j]).shift(1).reindex(entries).to_numpy()
                covariance[:, i, j] = covariance[:, j, i] = values
        return covariance

    def _allocate(self, levels):
        targets = pd.DataFrame(0., index=self.index_events.index, columns=self.traded_instruments)
        entries = self.index_events[self.index_events != 0]
        details = []
        columns = ["index_units", "actual_units", "index_deviation", "selected_count",
                   "selected_weight", "exante_risk_ratio", "risk_cap_hit"]
        for ccy in self.component_weights.index:
            columns += [f"deviation_{ccy}", f"contribution_{ccy}", f"selected_{ccy}", f"exposure_{ccy}"]
        if entries.empty:
            return targets, pd.DataFrame(columns=columns, index=entries.index.rename("timestamp"))
        moves = levels.diff(self._bars(self.volatility_horizon))
        covariance = self._entry_covariance(moves, entries.index)
        deviations = self.component_deviations.reindex(entries.index)
        w = self.component_weights.to_numpy()
        for k, (timestamp, q) in enumerate(entries.items()):
            d = deviations.iloc[k].to_numpy()
            contribution = w*d
            chosen = ((contribution*contribution.sum() > 0) if self.constituent_selection == "aligned"
                      else w > 0)
            allocation = w*chosen
            mass = allocation.sum()
            if not np.isfinite(d).all() or mass <= 0:
                raise ValueError("No finite aligned allocation at an index entry.")
            if self.sizing_mode != "retained":
                allocation /= mass
            cov = covariance[k]
            index_var, subset_var = w@cov@w, allocation@cov@allocation
            if not np.isfinite(cov).all() or min(index_var, subset_var) <= 0:
                raise ValueError("Non-positive or unavailable entry covariance.")
            requested = q*np.sqrt(index_var/subset_var) if self.sizing_mode == "risk_matched" else q
            actual = float(np.clip(requested, -self.max_units, self.max_units))
            exposures = actual*allocation
            targets.loc[timestamp] = exposures@self.cross_to_pair.to_numpy()
            row = dict(index_units=q, actual_units=actual, index_deviation=contribution.sum(),
                       selected_count=int(chosen.sum()), selected_weight=mass,
                       exante_risk_ratio=abs(actual/q)*np.sqrt(subset_var/index_var),
                       risk_cap_hit=abs(requested) > self.max_units)
            for ccy, dev, contrib, keep, exposure in zip(self.component_weights.index, d, contribution, chosen, exposures):
                row.update({f"deviation_{ccy}": dev, f"contribution_{ccy}": contrib,
                            f"selected_{ccy}": bool(keep), f"exposure_{ccy}": exposure})
            details.append(row)
        return targets, pd.DataFrame(details, index=entries.index.rename("timestamp"))

    def generate_signals(self) -> dict[str, pd.Series]:
        common_index, levels, index_level, spreads, full_spread = self._prepare()
        half_life = self._bars(self.equilibrium_half_life)
        minimum = self._bars(self.equilibrium_min_history)
        reference = index_level.ewm(halflife=half_life, min_periods=minimum).mean().shift(1)
        deviation = index_level-reference
        scale = deviation.ewm(halflife=self._bars(self.zscore_half_life),
                              min_periods=self._bars(self.zscore_min_history)).std().shift(1)
        moves = index_level.diff(self._bars(self.volatility_horizon))
        volatility = moves.ewm(halflife=self._bars(self.volatility_half_life),
                              min_periods=self._bars(self.volatility_min_history)).std().shift(1)
        self.features = pd.DataFrame({"level": index_level, "reference": reference, "deviation": deviation,
                                      "zscore": deviation/scale.replace(0, np.nan),
                                      "edge": -self.reversion_fraction*deviation,
                                      "spread": full_spread, "volatility": volatility})
        self.component_deviations = levels-levels.ewm(halflife=half_life, min_periods=minimum).mean().shift(1)
        self.index_events = self._index_requests()
        self.pair_targets, self.entry_details = self._allocate(levels)
        # Missing-quote exits are requests. Map to actual common quote times
        # before exposing targets on asynchronous original pair grids.
        self.common_signals = self.pair_targets.reindex(common_index, method="ffill").fillna(0.)
        self.signals = {pair: self.common_signals[pair].reindex(self.fx_price_dict[pair].index, method="ffill")
                        .fillna(0.).rename(pair) for pair in self.traded_instruments}
        return self.signals
