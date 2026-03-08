import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def plot_daily_recap(instrument, date_range, close_price_df, strategy, trader):

    start_ts = pd.Timestamp(
        date_range[0].year, date_range[0].month, date_range[0].day, 17, tz="US/Eastern"
    ) - pd.Timedelta(days=1)

    end_ts = pd.Timestamp(
        date_range[1].year, date_range[1].month, date_range[1].day, 17, tz="US/Eastern"
    )

    df = close_price_df.copy()
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    df["signal"] = strategy.generate_signals()[instrument].reindex(df.index).fillna(0)
    df["position"] = trader.generate_positions()[instrument].reindex(df.index).fillna(0)
    df["pnl"] = trader.generate_net_pnl()[instrument].reindex(df.index).fillna(0)
    df["cum_pnl"] = df["pnl"].cumsum()
    trades = trader.generate_trades()[instrument]
    trades = trades.loc[(trades.index >= start_ts) & (trades.index <= end_ts)]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.45, 0.2, 0.15, 0.2],
    )

    # Price
    fig.add_trace(
        go.Scatter(x=df.index, y=df["mid"], name="Price", line=dict(color="blue")),
        row=1,
        col=1,
    )

    # Position
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["position"], name="Position", line=dict(color="orange")
        ),
        row=2,
        col=1,
    )

    # Signals
    fig.add_trace(
        go.Scatter(x=df.index, y=df["signal"], name="Signal", line=dict(color="red")),
        row=3,
        col=1,
    )

    # Cumulative PnL
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["cum_pnl"], name="Cumulative PnL", line=dict(color="green")
        ),
        row=4,
        col=1,
    )

    # Trades on price subplot
    trade_markers = trades[trades["trade_size"].notna()]
    buy_markers = trade_markers[trade_markers["trade_size"] > 0]
    sell_markers = trade_markers[trade_markers["trade_size"] < 0]

    if not buy_markers.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_markers.index,
                y=buy_markers["trade_price"],
                mode="markers",
                marker=dict(color="green", size=10, symbol="triangle-up"),
                name="Buy",
                text=[
                    f"Size: {size}<br>Price: {price:.5f}<br>Cost: {cost:.5f}"
                    for size, price, cost in zip(
                        buy_markers["trade_size"],
                        buy_markers["trade_price"],
                        buy_markers["trade_cost"],
                    )
                ],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

    if not sell_markers.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_markers.index,
                y=sell_markers["trade_price"],
                mode="markers",
                marker=dict(color="red", size=10, symbol="triangle-down"),
                name="Sell",
                text=[
                    f"Size: {size}<br>Price: {price:.5f}<br>Cost: {cost:.5f}"
                    for size, price, cost in zip(
                        sell_markers["trade_size"],
                        sell_markers["trade_price"],
                        sell_markers["trade_cost"],
                    )
                ],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

    fig.update_layout(
        title=f"{instrument} Daily Recap - {start_ts.strftime('%Y-%m-%d %H:%M')} to {end_ts.strftime('%Y-%m-%d %H:%M')}",
        height=900,
    )
    fig.update_xaxes(title_text="Time", row=4, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Position", row=2, col=1)
    fig.update_yaxes(title_text="Signal", row=3, col=1)
    fig.update_yaxes(title_text="Cumulative PnL", row=4, col=1)
    return fig
