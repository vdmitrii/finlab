import pandas as pd
import numpy as np
from scipy.signal import lfiltic, lfilter
from numba import jit
# from tqdm import tqdm


def tick_bars(df: pd.DataFrame, col: str = 'number_of_trades', max_trades: int = 100) -> pd.DataSeries:
    """
    Compute Tick Bars

    Args:
        df (pd.DataFrame): pandas dataframe with data
        col (str): name of columns (number_of_trades)
        t (int): number of ticks for grouping

    Returns:
        pd.DataSeries: new series with tick_bars
    """
    s = df[col].values
    tick_bars_col = pd.Series([])
    num_of_trades = 0

    for idx, row in enumerate(s):
        num_of_trades += s[idx]
        if num_of_trades >= max_trades:
            tick_bars_col.append(df.iloc[idx])

    df['tick_bars'] = tick_bars_col

    return df


def volume_bars(df: pd.DataFrame, col: str = 'volume', max_traded_volume: int = 100) -> pd.DataSeries:
    """
    Compute Volue Bars

    Args:
        df (pd.DataFrame): pandas dataframe with data
        col (str): name of columns (number_of_trades)
        traded_volume (int): number of ticks for grouping

    Returns:
        pd.DataSeries: new series with tick_bars
    """
    s = df[col].values
    volume_bars_col = pd.Series([])
    traded_volume = 0

    for idx, row in enumerate(s):
        traded_volume += s[idx]
        if traded_volume >= max_traded_volume:
            volume_bars_col.append(df.iloc[idx])

    df['volume_bars'] = volume_bars_col

    return df


def dollar_bars(df: pd.DataFrame, asset_volume: str = 'quote_asset_volume', max_traded: int = 100) -> pd.DataSeries:
    """
    Compute Dollar Bars

    Args:
        df (pd.DataFrame): pandas dataframe with data
        asset_volume: (str): traded volume in quote asset
        max_traded (int): traded amount of dollars
    Returns:
        pd.DataSeries: new series with tick_bars
    """
    asset_volume = df[asset_volume].values
    dollar_bars_col = pd.Series([])
    traded_dollars = 0

    for idx, row in enumerate(asset_volume):
        traded_dollars += asset_volume[idx]
        if traded_dollars >= max_traded:
            dollar_bars_col.append(df.iloc[idx])

    df['dollar_bars'] = traded_dollars

    return df


def dollar_imbalance_bars(df: pd.DataFrame, price: str = 'price', volume: str = 'volume') -> pd.DataFrame:
    """Dollar Imbalance Bars

    1. Compute signed flows:
    1.1 Compute tick direction (the sign of change in price).
    1.2 Multiply tick direction by tick volume.

    2. Accumulate the imbalance bars :
    2.1 Starting from the first datapoint, step through the dataset and keep track of the cumulative signed flows (the imbalance).
    2.2 Take a sample whenever the absolute value of imbalance exceeds the expected imbalance threshold.
    3.3 Update the expectations of imbalance threshold as you see more data.

    Args:
        df (pd.DataFrame): input dataframe
        price (str, optional): close price. Defaults to 'price'.
        volume (str, optional): traded volume. Defaults to 'volume'.

    Returns:
        pd.DataFrame: final dataframe with added colomn
    """
    new_col = pd.Series([])

    delta_price = df.close.diff().fillna(0)

    for idx, row in df.iterrows():
        diff = df.loc[idx - 1, price]
        if row.diff() == 0:
            new_col.append(row.diff())
        elif row.diff() != 0:


def vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Volume weighted average price

    Args:
        df (pd.DataFrame): dataframe with price and volume columns

    Returns:
        pd.DataFrame: new dataframe with added column 'vmap'
    """
    q = df['volume']
    p = df['close']
    vwap = np.sum(q * p) / np.sum(q)
    df['vwap'] = vwap
    return df


def ewma(x: np.array, alpha: float) -> np.array:
    '''Exponentially weighted moving average

    Returns the exponentially weighted moving average of x.

    Args:
        x (np.array): array-like
        alpha (float): 0 <= alpha <= 1

    Returns:
        ewma(np.array): exponentially weighted moving average
    '''
    # Coerce x to an array
    x = np.array(x)
    n = x.size

    # Create an initial weight matrix of (1-alpha), and a matrix of powers
    # to raise the weights by
    w0 = np.ones(shape=(n, n)) * (1-alpha)
    p = np.vstack([np.arange(i, i-n, -1) for i in range(n)])

    # Create the weight matrix
    w = np.tril(w0**p, 0)

    # Calculate the ewma
    return np.dot(w, x[::np.newaxis]) / w.sum(axis=1)


def ewma_linear_filter(array: np.array, window: int):
    alpha = 2 / (window + 1)
    b = [alpha]
    a = [1, alpha-1]
    zi = lfiltic(b, a, array[0:1], [0])
    return lfilter(b, a, array, zi=zi)[0]


# Check it out later
@jit(nopython=True)
def numba_isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return np.fabs(a-b) <= np.fmax(rel_tol*np.fmax(np.fabs(a), np.fabs(b)), abs_tol)


@jit(nopython=True)
def bt(p0, p1, bs):
    # if math.isclose((p1 - p0), 0.0, abs_tol=0.001):
    if numba_isclose((p1-p0), 0.0, abs_tol=0.001):
        b = bs[-1]
        return b
    else:
        b = np.abs(p1-p0)/(p1-p0)
        return b


@jit(nopython=True)
def get_imbalance(t):
    bs = np.zeros_like(t)
    for i in np.arange(1, bs.shape[0]):
        t_bt = bt(t[i-1], t[i], bs[:i-1])
        bs[i-1] = t_bt
    return bs[:-1]  # remove last value
