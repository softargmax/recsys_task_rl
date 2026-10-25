import numpy as np

ALL = 1 << 40
TS_MASK = 0xFFFFFFFF
NEGLIGIBLE = 700


class GroupIndex:
    def __init__(self, group, ts, row=None):
        group = np.asarray(group, dtype=np.int64)
        if row is None:
            key = (group << 32) | np.asarray(ts, dtype=np.int64)
            order = np.argsort(key, kind="stable")
            self.key_ts = key[order]
            del key
            self.key_row = None
        else:
            ts = np.asarray(ts, dtype=np.int64)
            row = np.asarray(row, dtype=np.int64)
            order = np.lexsort((row, group))
            base = group[order] << 32
            self.key_ts = base | ts[order]
            self.key_row = base | row[order]
        self.order = order.astype(np.int32) if len(order) < 2**31 else order
        del order
        self.n = len(group)

    def cumsum(self, values):
        out = np.zeros(self.n + 1, dtype=np.float64)
        np.cumsum(np.asarray(values, dtype=np.float64)[self.order], out=out[1:])
        return out

    def sorted_values(self, values):
        return np.asarray(values)[self.order]

    def decay_prefixes(self, values, tau):
        return decay_prefixes(self, values, tau)

    def span(self, q_group, end_row=None, end_ts=None, inclusive=False):
        base = np.asarray(q_group, dtype=np.int64) << 32
        start = np.searchsorted(self.key_ts, base, side="left")
        if end_row is not None:
            hi = np.searchsorted(self.key_row, base | np.asarray(end_row, dtype=np.int64), side="left")
        else:
            hi = np.searchsorted(self.key_ts, base | np.asarray(end_ts, dtype=np.int64), side="right" if inclusive else "left")
        return start, np.maximum(hi, start)

    def window_start(self, q_group, start, hi, from_ts):
        base = np.asarray(q_group, dtype=np.int64) << 32
        lo = np.searchsorted(self.key_ts, base | np.maximum(np.asarray(from_ts, dtype=np.int64), 0), side="left")
        return np.clip(lo, start, hi)


class WindowQuery:
    def __init__(self, index, q_group, ref_ts, end_row=None, end_ts=None, inclusive=False, valid=None):
        self.index = index
        self.q_group = np.where(valid, q_group, 0) if valid is not None else np.asarray(q_group)
        self.valid = valid
        self.ref_ts = np.asarray(ref_ts, dtype=np.int64)
        self.start, self.hi = index.span(self.q_group, end_row=end_row, end_ts=end_ts, inclusive=inclusive)
        if valid is not None:
            self.hi = np.where(valid, self.hi, self.start)
        self._lo = {}

    def lo(self, window):
        if window >= ALL:
            return self.start
        if window not in self._lo:
            self._lo[window] = self.index.window_start(self.q_group, self.start, self.hi, self.ref_ts - window)
        return self._lo[window]

    def sum(self, csum, window):
        return (csum[self.hi] - csum[self.lo(window)]).astype(np.float32)

    def last(self, sorted_values, fill=np.nan):
        has = self.hi > self.start
        out = np.full(len(self.hi), fill, dtype=np.float64)
        out[has] = sorted_values[self.hi[has] - 1]
        return out

    def last_ts(self):
        has = self.hi > self.start
        out = np.full(len(self.hi), np.nan, dtype=np.float64)
        out[has] = self.index.key_ts[self.hi[has] - 1] & TS_MASK
        return out

    def first_ts(self):
        has = self.hi > self.start
        out = np.full(len(self.hi), np.nan, dtype=np.float64)
        out[has] = self.index.key_ts[self.start[has]] & TS_MASK
        return out

    def decayed(self, prefix):
        has = self.hi > self.start
        last = np.maximum(self.hi - 1, 0)
        log_sum = np.where(has, prefix.log_prefix[last] - prefix.offset[last] - self.ref_ts / prefix.tau, -np.inf)
        return self.masked(np.exp(log_sum)).astype(np.float32)

    def masked(self, values):
        if self.valid is None:
            return values
        return np.where(self.valid, values, np.nan)


class DecayPrefix:
    def __init__(self, log_prefix, offset, tau):
        self.log_prefix = log_prefix
        self.offset = offset
        self.tau = float(tau)


def decay_prefixes(index, values, tau):
    ts = (index.key_ts & TS_MASK).astype(np.float64)
    group_gap = ts.max(initial=0) / tau + NEGLIGIBLE
    group = index.key_ts >> 32
    offset = np.r_[0, np.cumsum(group[1:] != group[:-1])].astype(np.float64) * group_gap
    base = ts / tau + offset
    out = {}
    for name, v in values.items():
        with np.errstate(divide="ignore"):
            x = np.log(index.sorted_values(np.asarray(v, dtype=np.float64))) + base
        out[name] = DecayPrefix(np.logaddexp.accumulate(x), offset, tau)
    return out
