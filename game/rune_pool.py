class RunePool:
    def __init__(self, domain: str):
        self.domain = domain
        self.pool = 0
        self.per_turn = 1

    def channel(self):
        self.pool += self.per_turn

    def can_afford(self, rune_cost: int) -> bool:
        return self.pool >= rune_cost

    def spend(self, rune_cost: int):
        if not self.can_afford(rune_cost):
            raise ValueError(f"Insufficient runes: need {rune_cost}, have {self.pool}")
        self.pool -= rune_cost

    def __repr__(self):
        return f"RunePool({self.domain}: {self.pool})"
