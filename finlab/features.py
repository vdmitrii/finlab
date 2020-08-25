import pandas as pd
import numpy as np
from scipy.signal import lfiltic, lfilter
from numba import jit


def get_t_events(g_raw: pd.TimedeltaIndex, h: int) -> pd.DatetimeIndex:
    """THE SYMMETRIC CUSUM FILTER

    Variable S could be based on any of the features like structural
    break statistics, entropy, or market microstructure measurements.

    Args:
        g_raw (pd.TimedeltaIndex): the raw time series we wish to filter
        h (int): the threshold

    Returns:
        pd.DatetimeIndex: [description]
    """
    t_events = []
    s_pos = 0
    s_neg = 0
    diff = g_raw.diff()
    for i in diff.index[1:]:
        s_pos = max(0, s_pos + diff.loc[i])
        s_neg = min(0, s_neg + diff.loc[i])
        if s_neg <= h:
            s_neg = 0
            t_events.append(i)
        elif s_pos > h:
            s_pos = 0
            t_events.append(i)
    return pd.DatetimeIndex(t_events)
