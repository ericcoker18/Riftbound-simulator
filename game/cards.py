from dataclasses import dataclass, field
from typing import Optional


class Card:
    def __init__(
        self,
        name: str,
        cost: int,
        rune_cost: int,
        health: int,
        card_type: str = "Unit",
        supertype: str = "",
        domain: Optional[str] = None,
        weight: float = 1.0,
        max_copies: int = 3,
        tags: list = None,
        keywords: dict = None,
        ability: str = "",
        signature: bool = False,
        signature_legend: Optional[str] = None,
    ):
        self.name = name
        self.cost = cost
        self.rune_cost = rune_cost      # power/rune cost (NOT attack damage)
        self.health = health             # base Might (combat strength)
        self.card_type = card_type
        self.supertype = supertype       # "Champion" for champion units
        self.domain = domain             # "Fury", "Calm", "Order", etc.
        self.weight = weight
        self.max_copies = max_copies
        self.tags = tags or []
        self.keywords = keywords or {}
        self.ability = ability
        self.signature = signature                  # True if signature card
        self.signature_legend = signature_legend    # e.g. "Irelia" — only usable in that legend's deck

    @property
    def champion(self) -> bool:
        return self.supertype == "Champion"

    def has(self, keyword: str) -> bool:
        return keyword in self.keywords

    def keyword_value(self, keyword: str, default=0):
        return self.keywords.get(keyword, default)

    def copy(self):
        return Card(
            self.name, self.cost, self.rune_cost, self.health,
            self.card_type, self.supertype, self.domain,
            self.weight, self.max_copies,
            list(self.tags), dict(self.keywords), self.ability,
            self.signature, self.signature_legend,
        )

    def __repr__(self):
        return f"{self.name}(cost={self.cost}/{self.rune_cost}r, might={self.health}, domain={self.domain})"


@dataclass
class UnitInstance:
    """A live instance of a unit card on the board. Separates template from game state."""
    card: Card
    is_exhausted: bool = True        # units enter exhausted by default
    current_health: int = 0          # set to card.health on creation; tracks damage taken
    battlefield_id: int = -1         # -1 = base, 0-2 = battlefield index
    buffs: list = field(default_factory=list)  # list of int bonuses to Might

    def __post_init__(self):
        if self.current_health == 0:
            self.current_health = self.card.health

    @property
    def effective_might(self) -> int:
        """Base Might + keyword bonuses (Assault/Shield applied externally in engine)."""
        return self.card.health + sum(self.buffs)

    @property
    def is_alive(self) -> bool:
        return self.current_health > 0

    def has(self, keyword: str) -> bool:
        return self.card.has(keyword)

    def keyword_value(self, keyword: str, default=0):
        return self.card.keyword_value(keyword, default)

    def __repr__(self):
        status = "exhausted" if self.is_exhausted else "ready"
        return f"{self.card.name}(might={self.effective_might}, hp={self.current_health}, {status})"
