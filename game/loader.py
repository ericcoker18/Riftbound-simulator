import json
from game.cards import Card
from game.keywords import parse_keywords


DOMAIN_TAGS = {"Order", "Fury", "Chaos", "Body", "Calm", "Mind"}

# Officially banned cards in constructed play (as of March 31, 2026)
BANNED_CARDS = {
    "Draven, Vanquisher",      # Unit
    "Draven - Vanquisher",     # API name format
    "Scrapheap",               # Gear
    "Called Shot",              # Spell
    "Fight or Flight",         # Spell
}


def heuristic_weight(card: Card) -> float:
    """
    Estimate card value for weighted deck building.

    Weights are balanced so spells and gear compete fairly with units.
    A good spell should be weighted similarly to a good unit.
    """
    total_cost = card.cost + (card.rune_cost * 2)

    if card.card_type == "Unit":
        might = card.health
        efficiency = might / max(total_cost, 1)

        kw_bonus = 0.0
        kw_bonus += card.keyword_value("Assault") * 0.3
        kw_bonus += card.keyword_value("Shield") * 0.2
        kw_bonus += 0.5 if card.has("Tank") else 0
        kw_bonus += 0.3 if card.has("Ganking") else 0
        kw_bonus += card.keyword_value("Hunt") * 0.2
        kw_bonus += 0.4 if card.has("Deflect") else 0

        return max(0.1, efficiency + kw_bonus)

    elif card.card_type == "Spell":
        # Spells are valuable — removal, draw, protection, combat tricks
        # Base weight comparable to decent units
        ability = card.ability.lower()
        base = 1.5

        # Removal spells are premium
        if any(kw in ability for kw in ["deal", "kill", "destroy", "damage"]):
            base = 2.5
        # Draw spells are great
        elif "draw" in ability:
            base = 2.0
        # Protection is valuable
        elif any(kw in ability for kw in ["counter", "shield", "ready", "guardian"]):
            base = 2.0
        # Combat tricks
        elif any(kw in ability for kw in ["might", "buff", "punch"]):
            base = 1.8

        # Cheap spells are more flexible
        cost_factor = max(0.5, 3.0 / max(total_cost, 1))
        return max(0.1, base * min(cost_factor, 2.0))

    elif card.card_type == "Gear":
        # Gear provides persistent value — should be weighted well
        ability = card.ability.lower()
        base = 1.5

        # Equipment (attaches to units) is premium
        if "equip" in ability:
            base = 2.5
        # Protective gear
        elif any(kw in ability for kw in ["guardian", "zhonya", "shield"]):
            base = 2.5

        cost_factor = max(0.5, 3.0 / max(total_cost, 1))
        return max(0.1, base * min(cost_factor, 2.0))

    return 0.1


def load_card_pool(filepath="data/cards.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        raw_cards = json.load(f)

    # Filter out banned cards
    raw_cards = [d for d in raw_cards if d["name"] not in BANNED_CARDS]

    cards = []
    for d in raw_cards:
        # Extract domain: first domain tag found in the card's tags list
        domain = None
        for tag in d.get("tags", []):
            if tag in DOMAIN_TAGS:
                domain = tag
                break
        # Fallback: check the dedicated domain field if present
        if domain is None:
            domain = d.get("domain")

        keywords = parse_keywords(d.get("ability", ""))

        card = Card(
            name=d["name"],
            cost=d["cost"],
            rune_cost=d.get("rune_cost", d.get("power", 0)) or 0,
            health=d["health"],
            card_type=d.get("card_type", "Unit"),
            supertype=d.get("supertype", ""),
            domain=domain,
            max_copies=d.get("max_copies", 3),
            tags=d.get("tags", []),
            keywords=keywords,
            ability=d.get("ability", ""),
            signature=d.get("signature", False),
            signature_legend=d.get("signature_legend"),
        )
        card.weight = heuristic_weight(card)
        cards.append(card)

    return cards
