import random
from typing import Optional


class RNG:
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def randint(self, a, b):
        return self._rng.randint(a, b)

    def choice(self, seq):
        return self._rng.choice(seq)

    def shuffle(self, seq):
        self._rng.shuffle(seq)

    def randrange(self, *args):
        return self._rng.randrange(*args)

    def sample(self, population, k):
        return self._rng.sample(population, k)

    def random(self):
        return self._rng.random()
