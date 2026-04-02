"""
Rune system for Riftbound.

Each player has 12 rune slots split between their legend's two domains.
Default split is 6/6 but can be weighted (e.g. 8/4).
Rune slots determine how many runes of each domain you can spend per turn.
Runes refresh each turn (not cumulative).
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

        # Current available runes this turn (refresh each turn)
        self.available1 = 0
        self.available2 = 0

    def refresh(self):
        """Called at start of turn — runes reset to full."""
        self.available1 = self.max1
        self.available2 = self.max2

    def can_afford(self, rune_cost: int, domain: str = None) -> bool:
        """
        Check if the player can afford a rune cost.
        If domain is specified, checks that specific domain's pool.
        If domain is None, checks if either pool has enough.
        """
        if rune_cost <= 0:
            return True

        if domain == self.domain1:
            return self.available1 >= rune_cost
        elif domain == self.domain2:
            return self.available2 >= rune_cost
        else:
            # Generic rune cost — either domain works
            return self.available1 >= rune_cost or self.available2 >= rune_cost

    def spend(self, rune_cost: int, domain: str = None):
        """
        Spend runes. Prefers the domain with more available if unspecified.
        """
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

    def __repr__(self):
        return f"RunePool({self.domain1}:{self.available1}/{self.max1}, {self.domain2}:{self.available2}/{self.max2})"
