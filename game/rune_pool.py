"""
Rune system for Riftbound.

Each player has 12 rune slots split between their legend's two domains.
Default split is 6/6 but can be weighted (e.g. 8/4).

Each turn, the player channels 2 runes (3 for P2's first turn).
Channeling adds runes from your slots to your available pool.
Runes accumulate across turns — they don't refresh.
"""


class RunePool:
    def __init__(self, domain1: str, domain2: str = None, split: tuple = None):
        """
        domain1, domain2: the legend's two domains (e.g. "Fury", "Chaos")
        split: tuple of (domain1_count, domain2_count) summing to 12.
               Default is (6, 6).
        """
        self.domain1 = domain1
        self.domain2 = domain2 or domain1

        if split:
            self.max1, self.max2 = split
        else:
            self.max1, self.max2 = 6, 6

        # How many runes have been channeled so far (grows each turn)
        self.channeled1 = 0
        self.channeled2 = 0

        # Available runes to spend this turn
        self.available1 = 0
        self.available2 = 0

    def channel(self, count: int = 2):
        """
        Channel runes at start of turn. Distributes evenly between
        domains, preferring the one with more unchanneled slots.
        """
        for _ in range(count):
            # Channel into whichever domain has more remaining slots
            remaining1 = self.max1 - self.channeled1
            remaining2 = self.max2 - self.channeled2

            if remaining1 >= remaining2 and remaining1 > 0:
                self.channeled1 += 1
                self.available1 += 1
            elif remaining2 > 0:
                self.channeled2 += 1
                self.available2 += 1
            elif remaining1 > 0:
                self.channeled1 += 1
                self.available1 += 1
            # else: all 12 slots channeled, nothing to add

    def can_afford(self, rune_cost: int, domain: str = None) -> bool:
        if rune_cost <= 0:
            return True
        if domain == self.domain1:
            return self.available1 >= rune_cost
        elif domain == self.domain2:
            return self.available2 >= rune_cost
        else:
            return self.available1 >= rune_cost or self.available2 >= rune_cost

    def spend(self, rune_cost: int, domain: str = None):
        if rune_cost <= 0:
            return
        if domain == self.domain1:
            self.available1 -= rune_cost
        elif domain == self.domain2:
            self.available2 -= rune_cost
        else:
            # Spend from whichever pool has more
            if self.available1 >= self.available2 and self.available1 >= rune_cost:
                self.available1 -= rune_cost
            elif self.available2 >= rune_cost:
                self.available2 -= rune_cost
            elif self.available1 >= rune_cost:
                self.available1 -= rune_cost

    @property
    def pool(self):
        """Total available runes (for backwards compatibility)."""
        return self.available1 + self.available2

    @property
    def total_channeled(self):
        return self.channeled1 + self.channeled2

    def __repr__(self):
        return (f"RunePool({self.domain1}:{self.available1}/{self.channeled1}/{self.max1}, "
                f"{self.domain2}:{self.available2}/{self.channeled2}/{self.max2})")
