import pandas as pd
import numpy as np
from scipy.signal import lfiltic, lfilter
from numba import jit
from tqdm import tqdm


def tick_bars(df: pd.DataFrame, col: str = 'number_of_trades', max_trades: int = 250) -> pd.DataFrame:
    """
    Compute Tick Bars

    Args:
        df (pd.DataFrame): pandas dataframe with data
        col (str): name of columns (number_of_trades)
        max_trades (int): max number of trades for grouping

    Returns:
        pd.DataFrame: new dataframe with trade bars
    """
    tick_series = df[col]
    trades = 0
    idxs = []
    for idx, num_of_trades in enumerate(tqdm(tick_series)):
        trades += num_of_trades
        if trades >= max_trades:
            idxs.append(idx)
            trades = 0
    return df.iloc[idxs].drop_duplicates()


def volume_bars(df: pd.DataFrame, volume_column: str = 'volume', max_traded_volume: int = 50) -> pd.DataFrame:
    """
    Compute Volume Bars

    Args:
        df (pd.DataFrame): pandas dataframe with data
        volume_column (str): name of columns (number_of_trades)
        max_traded_volume (int): maximum trading volume for the tick

    Returns:
        pd.DataFrame: new dataframe with tick_bars
    """
    volume_series = df[volume_column]
    vol = 0
    idxs = []
    for idx, volume in enumerate(tqdm(volume_series)):
        vol += volume
        if vol >= max_traded_volume:
            idxs.append(idx)
            vol = 0
    return df.iloc[idxs].drop_duplicates()


def dollar_bars(df: pd.DataFrame, amount: str = 'amount', max_amount: int = 500000) -> pd.DataFrame:
    """
    Compute Dollar Bars

    Args:
        df (pd.DataFrame): pandas dataframe with data
        amount: (str): column with dollar amount
        max_amount (int): max amount of dollars
    Returns:
        pd.DataFrame: new dataframe with dollar bars
    """
    amount_series = df[amount]
    amnt = 0
    idxs = []
    for idx, amount in enumerate(tqdm(amount_series)):
        amnt += amount
        if amnt >= max_amount:
            idxs.append(idx)
    return df.iloc[idxs].drop_duplicates()


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

    pass


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


@jit(nopython=True)
def numba_isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return np.fabs(a-b) <= np.fmax(rel_tol*np.fmax(np.fabs(a), np.fabs(b)), abs_tol)


@jit(nopython=True)
def bt(p0, p1, bs):
    # if math.isclose((p1 - p0), 0.0, abs_tol=0.001):
    if numba_isclose((p1 - p0), 0.0, abs_tol=0.001):
        b = bs[-1]
        return b
    else:
        b = np.abs(p1 - p0) / (p1 - p0)
        return b


@jit(nopython=True)
def get_imbalance(t):
    bs = np.zeros_like(t)
    for i in np.arange(1, bs.shape[0]):
        t_bt = bt(t[i - 1], t[i], bs[:i - 1])
        bs[i - 1] = t_bt
    return bs[:-1]  # remove last value
