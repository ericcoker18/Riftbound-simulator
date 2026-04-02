"""
Legend and champion identity system.

Each Legend defines two domains (colors). A legal deck can only contain
cards from those two domains. Each Legend also has associated Champion
cards that can be included in the deck.
"""

import json

VALID_DOMAINS = frozenset({"Order", "Fury", "Chaos", "Body", "Calm", "Mind"})


class Legend:
    """
    Represents the Legend card that determines a deck's domain identity.
    In the real game this is a separate card zone. In simulation it is deck metadata.
    """
    def __init__(self, name: str, domains: list, champion_tag: str = ""):
        self.name = name
        self.domains = set(domains)          # e.g. {"Fury", "Chaos"}
        self.champion_tag = champion_tag     # e.g. "Draven"

    def is_legal(self, card) -> bool:
        """A card is legal if its domain is one of this legend's two domains."""
        return card.domain in self.domains

    def get_own_champions(self, card_pool) -> list:
        """Return champion cards that specifically belong to this legend."""
        return [
            c for c in card_pool
            if c.champion and self.champion_tag in c.tags
        ]

    def get_champions(self, card_pool) -> list:
        """Return all champion cards legal for this legend's two domains."""
        return [c for c in card_pool if c.champion and self.is_legal(c)]

    def get_legal_pool(self, card_pool) -> list:
        """Return all non-champion cards legal for this legend's domains."""
        return [c for c in card_pool if self.is_legal(c) and not c.champion]

    def __repr__(self):
        return f"Legend({self.name}, {'/'.join(sorted(self.domains))})"


def load_legends(filepath="data/legends.json") -> list:
    """Load all legends from JSON. Returns list of Legend objects."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    legends = []
    seen = set()
    for entry in raw:
        name = entry["name"]
        # Deduplicate (e.g. Master Yi has two legend variants with same domains)
        tag = entry.get("champion_tag", name.split(" - ")[0])
        if tag in seen:
            continue
        seen.add(tag)
        legends.append(Legend(
            name=name,
            domains=entry["domains"],
            champion_tag=tag,
        ))
    return legends
