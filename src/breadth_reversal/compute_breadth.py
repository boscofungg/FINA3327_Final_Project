import pandas as pd
import numpy as np


def compute_signal(df, diff_lag=1, ma_window=5, pct_window=20,
                   pct_threshold=10, holding_period=1):
    df = df.copy()

    # Step 1 — Breadth difference with configurable lag
    df['Breadth_Diff'] = df['Breadth'].diff(diff_lag)

    # Step 2 — Rolling MA of the difference
    df['Rolling_5D_Mean'] = df['Breadth_Diff'].rolling(window=ma_window).mean()

    # Step 3 — Rolling percentile threshold
    df['Percentile_10'] = df['Rolling_5D_Mean'].rolling(window=pct_window).quantile(pct_threshold / 100)

    # Step 4 — Raw signal
    df['Raw_Signal'] = (df['Rolling_5D_Mean'] < df['Percentile_10']).astype(int)

    # Step 5 — Holding period (expand signal forward N days)
    if holding_period == 1:
        df['Signal'] = df['Raw_Signal']
    else:
        df['Signal'] = df['Raw_Signal'].rolling(window=holding_period, min_periods=1).max().astype(int)

    df['Entry'] = ((df['Signal'] == 1) & (df['Signal'].shift(1) == 0)).astype(int)
    df['Exit']  = ((df['Signal'] == 0) & (df['Signal'].shift(1) == 1)).astype(int)

    # Bucket analysis (always window=20 so bucket plots are comparable
    # across sensitivity runs with different pct_window values).
    def rolling_percentile_rank(series, window=20):
        return series.rolling(window=window).apply(
            lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (window - 1) * 100
        )

    df['Rolling_Percentile'] = rolling_percentile_rank(df['Rolling_5D_Mean'], window=20)
    df['Bucket'] = pd.cut(df['Rolling_Percentile'],
                          bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                          labels=['0-10', '10-20', '20-30', '30-40', '40-50',
                                  '50-60', '60-70', '70-80', '80-90', '90-100'],
                          include_lowest=True)

    df.dropna(inplace=True)
    return df
