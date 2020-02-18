class EpsilonExpDecay:
    def __init__(self, start, end, rate):
        self._start = start
        self._eps = start
        self._rate = rate
        self._end = end

    @property
    def epsilon(self):
        return self._eps

    def update(self):
        self._eps = max(self._eps * self._rate, self._end)

    def __repr__(self):
        return f"EpsilonExpDecay(start={self._start}, end={self._end}, rate={self._rate})"


class GaussianNoiseDecay:
    def __init__(self, mu, start_sigma, end_sigma, rate):
        pass

    @property
    def epsilon(self):
        return 1.0

    def update(self):
        pass
