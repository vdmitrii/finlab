import pandas as pd
import numpy as np
from scipy.signal import lfiltic, lfilter
from numba import jit


def get_daily_vol(close: int, span0: int = 100):
    """Daily volatility

    Daily volatility, reindexed to close

    Args:
        close ([type]): [description]
        span0 (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # d returns
    df0 = df0.ewm(span=span0).std()
    return df0


def apply_TP_Sl_on_t1(close: pd.Series, events: pd.DataFrame, ptSl: list, molecule: list) -> pd.DataFrame:
    """Triple-Barrier Labeling Method

    Args:
        close (pd.Series): A pandas series of prices
        events (pd.DataFrame): A pandas dataframe, with columns
        t1 (type): The timestamp of vertical barrier. When the value is np.nan,
            there will not be a vertical barrier.
        trgt (type): The unit width of the horizontal barriers.
        ptSl (list): A list of two non-negative float values [ pt,sl,t1 ]:
        ptSl[0] (list): The factor that multiplies trgt to set the width of the upper barrier.
            If 0, there will not be an upper barrier.
        ptSl[1] (list): The factor that multiplies trgt to set the width of the lower barrier.
            If 0, there will not be a lower barrier.
        molecule (list): A list with the subset of event indices that will be processed by a
            single thread. Its use will become clear later on in the chapter.

    Returns:
        pd.DataFrame: The output from this function is a pandas dataframe containing the timestamps
            (if any) at which each barrier was touched.
    """
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index)  # NaNs
    if ptSl[1] > 0:
        sl = -ptSl[1]*events_['trgt']
    else:
        sl = pd.Series(index=events.index)  # NaNs
    for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1]  # path prices
        df0 = (df0/close[loc]-1)*events_.at[loc, 'side']  # path returns
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()  # earliest SL
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()  # earliest TP
    return out


def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    # 1) get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]  # minRet
    # 2) get t1 (max holding period)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)
    # 3) form events object, apply stop loss on t1
    if side is None:
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2]
    events = (pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1)
                .dropna(subset=['trgt']))
    df0 = mpPandasObj(func=applyPtSlOnT1, pdObj=('molecule', events.index),
                numThreads=numThreads, close=close, events=events, ptSl=ptSl_)
    events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores nan
    if side is None:
        events = events.drop('side', axis=1)
    return events


def addVerticalBarrier(tEvents, close, numDays=1):
    t1 = close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
    t1 = t1[t1 < close.shape[0]]
    t1 = (pd.Series(close.index[t1], index=tEvents[:t1.shape[0]]))
    return t1
