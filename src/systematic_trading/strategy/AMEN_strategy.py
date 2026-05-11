from strategy.strategy import Strategy
from util.math_util import mad_clip

import math
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm


class AmenStrategy(Strategy):
    def __init__(
        self,
        traded_instruments: tuple[str],
        fx_price_dict: dict[str, pd.DataFrame],
        equity_close_dict: dict[str, pd.DataFrame],
        bond_close_dict: dict[str, pd.DataFrame],
        hyper_param_dict: dict | None = None,
    ):
        if hyper_param_dict is None:
            hyper_param_dict = {}

        look_back_months = hyper_param_dict.get("look_back_months", 60)
        if look_back_months <= 0:
            raise ValueError("look_back_months must be positive.")
        prediction_rank_threshold = hyper_param_dict.get(
            "prediction_rank_threshold", 1 / 3
        )
        if not 0 < prediction_rank_threshold <= 1:
            raise ValueError("prediction_rank_threshold must be in (0, 1].")
        prediction_zscore_threshold = hyper_param_dict.get(
            "prediction_zscore_threshold", 0.0
        )
        if prediction_zscore_threshold < 0:
            raise ValueError("prediction_zscore_threshold must be non-negative.")
        use_all_countries = hyper_param_dict.get("use_all_countries", True)

        self.instruments = traded_instruments
        self.fx_price_dict = fx_price_dict
        self.equity_close_dict = equity_close_dict
        self.bond_close_dict = bond_close_dict
        self.entry_time_offset_from_month_end = pd.Timedelta(
            hyper_param_dict.get(
                "entry_time_offset_from_month_end", pd.Timedelta(hours=2)
            )
        )
        self.exit_time_offset_from_month_end = pd.Timedelta(
            hyper_param_dict.get(
                "exit_time_offset_from_month_end", pd.Timedelta(hours=0)
            )
        )
        self.look_back_months = look_back_months
        self.prediction_rank_threshold = float(prediction_rank_threshold)
        self.prediction_zscore_threshold = float(prediction_zscore_threshold)
        self.use_all_countries = bool(use_all_countries)
        sizing_option = hyper_param_dict.get(
            "sizing_option",
            "binary"
        )
        if not sizing_option in ["binary", "zscore"]:
            raise ValueError("sizing_option must be either 'binary' or 'zscore'." \
            "")
        self.sizing_option = sizing_option
        self.regression_log_path = Path(
            hyper_param_dict.get(
                "regression_log_path", "amen_regression_diagnostics.csv"
            )
        )

    @staticmethod
    def _month_end_time(period: pd.Period) -> pd.Timestamp:
        month_end_date = period.asfreq("D", how="end").to_timestamp()
        while month_end_date.dayofweek >= 5:
            month_end_date -= pd.Timedelta(days=1)
        return pd.Timestamp(
            month_end_date.year,
            month_end_date.month,
            month_end_date.day,
            11,
            tz="US/Eastern",
        )

    @staticmethod
    def _asof_index_and_value(
        prices: pd.Series, ts: pd.Timestamp
    ) -> tuple[pd.Timestamp, float] | tuple[None, None]:
        pos = prices.index.searchsorted(ts, side="right") - 1
        if pos < 0:
            return None, None
        return prices.index[pos], prices.iloc[pos]

    @staticmethod
    def _subtract_weekend_excluding_offset(
        ts: pd.Timestamp, offset: pd.Timedelta
    ) -> pd.Timestamp:
        current = pd.Timestamp(ts)
        remaining = pd.Timedelta(offset)

        while remaining > pd.Timedelta(0):
            saturday_start = current.normalize() - pd.Timedelta(
                days=max(current.dayofweek - 5, 0)
            )
            if current.dayofweek == 6 or (
                current.dayofweek == 5 and current != saturday_start
            ):
                current = saturday_start
                continue

            business_week_start = current.normalize() - pd.Timedelta(
                days=min(current.dayofweek, 5)
            )
            available = current - business_week_start

            if remaining <= available:
                return current - remaining

            remaining -= available
            current = business_week_start - pd.Timedelta(days=2)

        return current

    @staticmethod
    def _instrument_currencies(instrument: str) -> tuple[str, str]:
        if len(instrument) < 6:
            raise ValueError(
                f"Cannot infer base and term currencies from instrument '{instrument}'."
            )
        return instrument[:3].upper(), instrument[3:6].upper()

    def _asset_columns_for_instrument(
        self, instrument: str, asset_columns: list[str]
    ) -> list[str]:
        if self.use_all_countries:
            return asset_columns

        base_ccy, term_ccy = self._instrument_currencies(instrument)
        relevant_ccys = {base_ccy, term_ccy}
        
        # always include US
        if not "USD" in relevant_ccys:
            relevant_ccys.add("USD")

        return [
            column
            for column in asset_columns
            if column.rsplit("_", maxsplit=1)[-1].upper() in relevant_ccys
        ]

    def create_month_end_fx_returns(
        self, time_offset_from_month_end: pd.Timedelta = pd.Timedelta(days=1)
    ) -> dict[str, pd.Series]:
        time_offset_from_month_end = pd.Timedelta(time_offset_from_month_end)
        if time_offset_from_month_end < pd.Timedelta(0):
            raise ValueError("time_offset_from_month_end must be non-negative.")

        fx_returns = {}

        for instrument in self.instruments:
            if instrument not in self.fx_price_dict:
                raise ValueError(f"Missing fx data for instrument '{instrument}'.")

            df = self.fx_price_dict[instrument]
            if "mid" not in df.columns:
                raise ValueError(
                    f"Expected a 'mid' column for fx instrument '{instrument}'."
                )

            prices = df["mid"].dropna().sort_index()
            prices = prices[~prices.index.duplicated(keep="last")]
            if prices.empty:
                fx_returns[instrument] = pd.Series(dtype=float, name=instrument)
                continue

            months = prices.index.tz_localize(None).to_period("M").unique().sort_values()
            monthly_rows = []

            for month in months:
                end_ts = self._month_end_time(month)
                start_ts = self._subtract_weekend_excluding_offset(
                    end_ts, time_offset_from_month_end
                )

                _, start_price = self._asof_index_and_value(prices, start_ts)
                _, end_price = self._asof_index_and_value(prices, end_ts)

                if start_price is None or end_price is None:
                    continue

                monthly_rows.append(
                    {"timestamp": start_ts, instrument: end_price / start_price - 1}
                )

            if not monthly_rows:
                fx_returns[instrument] = pd.Series(dtype=float, name=instrument)
                continue

            instrument_returns = pd.DataFrame(monthly_rows).set_index("timestamp")[
                instrument
            ]
            instrument_returns.name = instrument
            fx_returns[instrument] = instrument_returns.sort_index()

        return fx_returns

    def create_monthly_asset_returns(
        self, time_offset_from_month_end: pd.Timedelta = pd.Timedelta(days=1)
    ) -> pd.DataFrame:
        time_offset_from_month_end = pd.Timedelta(time_offset_from_month_end)
        if time_offset_from_month_end < pd.Timedelta(0):
            raise ValueError("time_offset_from_month_end must be non-negative.")

        def build_asset_returns(
            close_dict: dict[str, pd.DataFrame], asset_name: str
        ) -> pd.DataFrame:
            asset_returns = {}

            for ccy, df in close_dict.items():
                if "close" not in df.columns:
                    raise ValueError(
                        f"Expected a 'close' column for {asset_name} {ccy} data."
                    )

                close = df["close"].dropna().sort_index()
                close = close[~close.index.duplicated(keep="last")]
                if close.empty:
                    continue

                monthly_rows = []
                month_periods = close.index.tz_localize(None).to_period("M")
                months = month_periods.unique().sort_values()

                for i in range(1, len(months)):
                    prev_month = months[i - 1]
                    month = months[i]

                    target_ts = self._subtract_weekend_excluding_offset(
                        self._month_end_time(month), time_offset_from_month_end
                    )

                    target_value_ts, target_close = self._asof_index_and_value(
                        close, target_ts
                    )
                    prev_month_close = close[month_periods == prev_month]

                    if prev_month_close.empty or target_close is None:
                        continue

                    prev_close = prev_month_close.iloc[-1]
                    if target_value_ts.tz_localize(None).to_period("M") != month:
                        continue

                    monthly_rows.append(
                        {
                            "timestamp": target_ts,
                            f"{asset_name}_{ccy}": target_close / prev_close - 1,
                        }
                    )

                if not monthly_rows:
                    continue

                ccy_returns = pd.DataFrame(monthly_rows).set_index("timestamp")

                if not ccy_returns.empty:
                    asset_returns[f"{asset_name}_{ccy}"] = ccy_returns.iloc[:, 0]

            if not asset_returns:
                return pd.DataFrame()

            return pd.DataFrame(asset_returns).sort_index()

        equity_returns = build_asset_returns(self.equity_close_dict, "equity")
        bond_returns = build_asset_returns(self.bond_close_dict, "bond")

        if equity_returns.empty:
            return bond_returns
        if bond_returns.empty:
            return equity_returns

        return pd.concat([equity_returns, bond_returns], axis=1).sort_index()

    def generate_signals(self) -> dict[str, pd.Series]:
        asset_past_return = self.create_monthly_asset_returns(
            self.entry_time_offset_from_month_end
        )
        fx_forward_return_dict = self.create_month_end_fx_returns(
            self.entry_time_offset_from_month_end
        )

        signals: dict[str, pd.Series] = {}
        asset_columns = list(asset_past_return.columns)
        regression_rows: list[dict[str, float | int | str]] = []

        for instrument in self.instruments:
            price_df = self.fx_price_dict.get(instrument)
            if price_df is None:
                raise ValueError(f"Missing fx data for instrument '{instrument}'.")

            signal = pd.Series(0.0, index=price_df.sort_index().index, name=instrument)
            fx_forward_return = fx_forward_return_dict.get(instrument)
            selected_asset_columns = self._asset_columns_for_instrument(
                instrument, asset_columns
            )

            if (
                fx_forward_return is None
                or fx_forward_return.empty
                or not selected_asset_columns
            ):
                signals[instrument] = signal
                continue

            combined = pd.concat(
                [
                    fx_forward_return.rename(instrument),
                    asset_past_return[selected_asset_columns],
                ],
                axis=1,
                join="inner",
            ).dropna()

            if combined.empty:
                signals[instrument] = signal
                continue

            for prediction_ts in combined.index:
                history = combined.loc[combined.index < prediction_ts].tail(
                    self.look_back_months
                )
                if len(history) < self.look_back_months:
                    continue

                y_train = history[instrument].astype(float)
                X_train = history[selected_asset_columns].astype(float)

                y_train = mad_clip(y_train, 2.0)
                X_train = mad_clip(X_train, 2.0)
                model = sm.OLS(y_train, X_train).fit()

                X_current = combined.loc[[prediction_ts], selected_asset_columns].astype(
                    float
                )
                predicted_return = float(model.predict(X_current).iloc[0])
                realized_fx_return = float(combined.loc[prediction_ts, instrument])
                history_predicted_returns = pd.Series(
                    model.predict(X_train), index=history.index
                )
                absolute_predictions = pd.concat(
                    [
                        history_predicted_returns.abs(),
                        pd.Series([abs(predicted_return)], index=[prediction_ts]),
                    ]
                )
                absolute_prediction_rank = int(
                    absolute_predictions.rank(
                        method="min", ascending=False
                    ).loc[prediction_ts]
                )
                rank_cutoff = math.ceil(
                    len(absolute_predictions) * self.prediction_rank_threshold
                )
                history_prediction_mean = float(history_predicted_returns.mean())
                history_prediction_std = float(history_predicted_returns.std())
                if pd.isna(history_prediction_std) or history_prediction_std == 0.0:
                    prediction_zscore = (
                        0.0
                        if predicted_return == history_prediction_mean
                        else float("inf")
                    )
                else:
                    prediction_zscore = (
                        predicted_return - history_prediction_mean
                    ) / history_prediction_std
                keep_signal = absolute_prediction_rank <= rank_cutoff
                keep_signal = keep_signal and (
                    abs(prediction_zscore) >= self.prediction_zscore_threshold
                )

                month_end_ts = self._month_end_time(
                    prediction_ts.tz_localize(None).to_period("M")
                )
                regression_row = {
                    "instrument": instrument,
                    "prediction_ts": prediction_ts.isoformat(),
                    "month_end_ts": month_end_ts.isoformat(),
                    "n_obs": len(history),
                    "r_squared": float(model.rsquared),
                    "adj_r_squared": float(model.rsquared_adj),
                    "predicted_return": predicted_return,
                    "realized_fx_return": realized_fx_return,
                    "absolute_prediction_rank": absolute_prediction_rank,
                    "rank_cutoff": rank_cutoff,
                    "prediction_zscore": float(prediction_zscore),
                    "prediction_rank_threshold": self.prediction_rank_threshold,
                    "prediction_zscore_threshold": self.prediction_zscore_threshold,
                    "use_all_countries": self.use_all_countries,
                    "regressor_columns": ",".join(selected_asset_columns),
                    "signal_kept": keep_signal,
                }
                regression_row.update(
                    {
                        f"coef_{column}": float(model.params.get(column, float("nan")))
                        for column in selected_asset_columns
                    }
                )
                regression_rows.append(regression_row)

                sized_signal = np.sign(predicted_return) if self.sizing_option == "binary" else prediction_zscore
                signal.loc[
                    (signal.index >= prediction_ts) & (signal.index < month_end_ts + self.exit_time_offset_from_month_end)
                ] = sized_signal if keep_signal else 0.0

            signals[instrument] = signal

        pd.DataFrame(regression_rows).to_csv(self.regression_log_path, index=False)

        return signals
