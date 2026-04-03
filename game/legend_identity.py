"""
Legend identity and deck building preferences.

Each legend has a specific game plan that should be reflected in
how decks are built. A Jhin deck should be spell-heavy (4-cost spells),
an Irelia deck should have cheap spells + movement, a Garen deck
should ramp into big units, etc.

These preferences influence:
  - Card weight multipliers during deck building (GA mutation/generation)
  - Minimum spell/gear counts to enforce archetype identity
  - Preferred cost curve shape
"""


# ---------------------------------------------------------------------------
# Legend deck building profiles
# ---------------------------------------------------------------------------

# Format: {
#   "min_spells": int,        # minimum spell count in deck
#   "min_gear": int,          # minimum gear count in deck
#   "max_units": int,         # cap on units to force spell/gear inclusion
#   "preferred_costs": [int], # card costs this legend wants to play
#   "spell_weight": float,    # multiplier on spell weights (>1 = prefer spells)
#   "gear_weight": float,     # multiplier on gear weights
#   "unit_weight": float,     # multiplier on unit weights (<1 = fewer units)
#   "cost_bonus": dict,       # {cost: bonus_weight} for preferred mana costs
#   "keyword_bonus": dict,    # {keyword: bonus} for preferred keywords
#   "description": str,       # what this legend wants to do
# }

LEGEND_PROFILES = {
    "Jhin": {
        "min_spells": 12,
        "min_gear": 2,
        "max_units": 22,
        "preferred_costs": [4],
        "spell_weight": 2.5,
        "gear_weight": 1.5,
        "unit_weight": 0.7,
        "cost_bonus": {4: 3.0, 3: 1.5, 5: 1.2},
        "keyword_bonus": {},
        "description": "Spell-focused: wants 4-cost spells to trigger legend ability",
    },
    "Irelia": {
        "min_spells": 14,
        "min_gear": 2,
        "max_units": 20,
        "preferred_costs": [1, 2],
        "spell_weight": 2.5,
        "gear_weight": 1.5,
        "unit_weight": 0.6,
        "cost_bonus": {1: 2.5, 2: 2.0},
        "keyword_bonus": {"Ganking": 3.0, "Deflect": 2.0},
        "description": "Cheap spells + movement: jump between battlefields with low-cost spells",
    },
    "Garen": {
        "min_spells": 6,
        "min_gear": 3,
        "max_units": 28,
        "preferred_costs": [4, 5, 6],
        "spell_weight": 1.2,
        "gear_weight": 1.5,
        "unit_weight": 1.0,
        "cost_bonus": {5: 1.5, 6: 1.5, 4: 1.2},
        "keyword_bonus": {},
        "description": "Ramp: build up to big units and overwhelm",
    },
    "Draven": {
        "min_spells": 8,
        "min_gear": 3,
        "max_units": 25,
        "preferred_costs": [1, 2, 3],
        "spell_weight": 1.5,
        "gear_weight": 2.0,
        "unit_weight": 0.9,
        "cost_bonus": {1: 1.5, 2: 1.5, 3: 1.3},
        "keyword_bonus": {"Assault": 2.0},
        "description": "Aggressive: cheap units + gear (Spinning Axe) for fast damage",
    },
    "Lux": {
        "min_spells": 14,
        "min_gear": 2,
        "max_units": 20,
        "preferred_costs": [3, 4, 5, 6],
        "spell_weight": 2.5,
        "gear_weight": 1.0,
        "unit_weight": 0.6,
        "cost_bonus": {6: 2.0, 5: 1.5, 4: 1.3, 3: 1.2},
        "keyword_bonus": {},
        "description": "Control: removal-heavy spell deck, Final Spark finisher",
    },
    "Ezreal": {
        "min_spells": 14,
        "min_gear": 0,
        "max_units": 22,
        "preferred_costs": [1, 2, 3],
        "spell_weight": 2.5,
        "gear_weight": 0.8,
        "unit_weight": 0.6,
        "cost_bonus": {1: 2.0, 2: 2.0, 3: 1.5},
        "keyword_bonus": {},
        "description": "Spell storm: cheap spells, draw engine, Ezreal chip damage",
    },
    "Fiora": {
        "min_spells": 8,
        "min_gear": 5,
        "max_units": 24,
        "preferred_costs": [2, 3],
        "spell_weight": 1.5,
        "gear_weight": 2.5,
        "unit_weight": 0.8,
        "cost_bonus": {2: 1.5, 3: 1.5},
        "keyword_bonus": {},
        "description": "Equipment: gear up champions with B.F. Sword, Doran's Blade",
    },
    "Sett": {
        "min_spells": 6,
        "min_gear": 4,
        "max_units": 26,
        "preferred_costs": [2, 3, 4],
        "spell_weight": 1.2,
        "gear_weight": 2.0,
        "unit_weight": 1.0,
        "cost_bonus": {3: 1.5, 4: 1.3},
        "keyword_bonus": {},
        "description": "Brawler: strong units + equipment, combat-focused",
    },
    "Viktor": {
        "min_spells": 12,
        "min_gear": 2,
        "max_units": 22,
        "preferred_costs": [2, 3, 4],
        "spell_weight": 2.0,
        "gear_weight": 1.5,
        "unit_weight": 0.7,
        "cost_bonus": {3: 1.5, 4: 1.3, 2: 1.3},
        "keyword_bonus": {},
        "description": "Control: removal + value generation over time",
    },
    "Ornn": {
        "min_spells": 4,
        "min_gear": 8,
        "max_units": 24,
        "preferred_costs": [3, 4],
        "spell_weight": 0.8,
        "gear_weight": 3.0,
        "unit_weight": 0.8,
        "cost_bonus": {4: 1.5, 3: 1.3},
        "keyword_bonus": {},
        "description": "Forge: gear-heavy deck, Ornn's signature equipment",
    },
    "Lee Sin": {
        "min_spells": 12,
        "min_gear": 2,
        "max_units": 22,
        "preferred_costs": [1, 2, 3],
        "spell_weight": 2.0,
        "gear_weight": 1.5,
        "unit_weight": 0.7,
        "cost_bonus": {1: 2.0, 2: 1.5, 3: 1.3},
        "keyword_bonus": {},
        "description": "Spell combo: cheap spells to trigger abilities + protection",
    },
    "Ahri": {
        "min_spells": 10,
        "min_gear": 3,
        "max_units": 24,
        "preferred_costs": [1, 2, 3],
        "spell_weight": 2.0,
        "gear_weight": 1.5,
        "unit_weight": 0.8,
        "cost_bonus": {1: 1.5, 2: 1.5, 3: 1.3},
        "keyword_bonus": {},
        "description": "Tempo: cheap spells + unit swaps for board advantage",
    },
    "Teemo": {
        "min_spells": 12,
        "min_gear": 2,
        "max_units": 22,
        "preferred_costs": [1, 2, 3],
        "spell_weight": 2.5,
        "gear_weight": 1.0,
        "unit_weight": 0.6,
        "cost_bonus": {1: 2.5, 2: 2.0, 3: 1.5},
        "keyword_bonus": {},
        "description": "Tricky: cheap spells, draw, hidden plays",
    },
    "Miss Fortune": {
        "min_spells": 8,
        "min_gear": 2,
        "max_units": 26,
        "preferred_costs": [3, 4, 5],
        "spell_weight": 1.5,
        "gear_weight": 1.2,
        "unit_weight": 1.0,
        "cost_bonus": {4: 1.5, 5: 1.5, 3: 1.2},
        "keyword_bonus": {},
        "description": "Midrange: ramp into big threats + Bullet Time",
    },
    "Poppy": {
        "min_spells": 8,
        "min_gear": 4,
        "max_units": 24,
        "preferred_costs": [1, 2, 3],
        "spell_weight": 1.5,
        "gear_weight": 2.0,
        "unit_weight": 0.9,
        "cost_bonus": {1: 1.5, 2: 1.5, 3: 1.3},
        "keyword_bonus": {},
        "description": "Wide board: cheap units + gear, swarm the battlefields",
    },
    "Yasuo": {
        "min_spells": 12,
        "min_gear": 2,
        "max_units": 22,
        "preferred_costs": [1, 2, 3],
        "spell_weight": 2.0,
        "gear_weight": 1.5,
        "unit_weight": 0.7,
        "cost_bonus": {1: 2.0, 2: 1.5, 3: 1.3},
        "keyword_bonus": {"Ganking": 3.0},
        "description": "Movement + spells: move between battlefields with cheap spells",
    },
    "Leona": {
        "min_spells": 8,
        "min_gear": 4,
        "max_units": 24,
        "preferred_costs": [2, 3, 4],
        "spell_weight": 1.5,
        "gear_weight": 2.0,
        "unit_weight": 0.9,
        "cost_bonus": {3: 1.5, 4: 1.3, 2: 1.2},
        "keyword_bonus": {"Shield": 2.0},
        "description": "Defensive: shields, gear, hold battlefields",
    },
}

# Default profile for legends not listed above
DEFAULT_PROFILE = {
    "min_spells": 8,
    "min_gear": 3,
    "max_units": 26,
    "preferred_costs": [],
    "spell_weight": 1.5,
    "gear_weight": 1.5,
    "unit_weight": 0.9,
    "cost_bonus": {},
    "keyword_bonus": {},
    "description": "Balanced: mix of units, spells, and gear",
}


def get_legend_profile(legend_name: str) -> dict:
    """Get the deck building profile for a legend."""
    # Extract short name (e.g. "Jhin" from "Jhin - Virtuoso")
    short = legend_name.split(" - ")[0].strip()
    return LEGEND_PROFILES.get(short, DEFAULT_PROFILE)


def apply_legend_weights(card, legend_name: str) -> float:
    """
    Get a weight multiplier for a card based on the legend's preferences.
    Returns a float multiplier (>1 = preferred, <1 = deprioritized).
    """
    profile = get_legend_profile(legend_name)
    multiplier = 1.0

    # Signature cards are inherently strong in their legend's deck
    # They were designed specifically for this legend — always include them
    if getattr(card, 'signature', False):
        sig_legend = getattr(card, 'signature_legend', None)
        legend_short = legend_name.split(" - ")[0].strip()
        if sig_legend == legend_short:
            multiplier *= 5.0  # very high priority — almost guaranteed inclusion
            return multiplier

    # Card type preference
    if card.card_type == "Spell":
        multiplier *= profile["spell_weight"]
    elif card.card_type == "Gear":
        multiplier *= profile["gear_weight"]
    elif card.card_type == "Unit":
        multiplier *= profile["unit_weight"]

    # Cost preference
    if card.cost in profile.get("cost_bonus", {}):
        multiplier *= profile["cost_bonus"][card.cost]

    # Keyword preference
    for kw, bonus in profile.get("keyword_bonus", {}).items():
        if card.has(kw):
            multiplier *= bonus

    return multiplier


def enforce_deck_composition(genome_cards, legend_name, card_pool):
    """
    Enforce minimum spell/gear counts for a legend's archetype.

    If the deck has too many units and not enough spells/gear,
    swap out the weakest units for the best available spells/gear.

    Returns a new card list (may be modified).
    """
    from collections import Counter

    profile = get_legend_profile(legend_name)
    counts = Counter(genome_cards)
    pool_lookup = {c.name: c for c in card_pool}

    # Count by type
    units = sum(count for name, count in counts.items()
                if pool_lookup.get(name) and pool_lookup[name].card_type == "Unit")
    spells = sum(count for name, count in counts.items()
                 if pool_lookup.get(name) and pool_lookup[name].card_type == "Spell")
    gear = sum(count for name, count in counts.items()
               if pool_lookup.get(name) and pool_lookup[name].card_type == "Gear")

    cards = list(genome_cards)  # mutable copy
    swaps_needed = 0

    # Check minimums
    spell_deficit = max(0, profile["min_spells"] - spells)
    gear_deficit = max(0, profile["min_gear"] - gear)
    unit_excess = max(0, units - profile["max_units"])

    swaps_needed = max(spell_deficit + gear_deficit, unit_excess)

    if swaps_needed == 0:
        return cards

    # Find units to swap out (lowest weight first)
    unit_entries = []
    for name, count in counts.items():
        card = pool_lookup.get(name)
        if card and card.card_type == "Unit" and not card.champion:
            unit_entries.append((card.weight, name, count))
    unit_entries.sort()  # lowest weight first

    # Find available spells/gear to swap in
    from game.legend import Legend, load_legends
    try:
        legends = load_legends()
        legend_obj = next(l for l in legends if l.name == legend_name)
    except (StopIteration, Exception):
        return cards

    available_spells = [
        c for c in card_pool
        if c.card_type == "Spell" and legend_obj.is_legal(c)
        and counts[c.name] < c.max_copies
    ]
    available_gear = [
        c for c in card_pool
        if c.card_type == "Gear" and legend_obj.is_legal(c)
        and counts[c.name] < c.max_copies
    ]

    # Sort by weight * legend preference
    available_spells.sort(
        key=lambda c: c.weight * apply_legend_weights(c, legend_name), reverse=True
    )
    available_gear.sort(
        key=lambda c: c.weight * apply_legend_weights(c, legend_name), reverse=True
    )

    # Perform swaps
    swap_count = 0
    spell_idx = 0
    gear_idx = 0

    for weight, unit_name, count in unit_entries:
        if swap_count >= swaps_needed:
            break

        # Remove one copy of this unit
        if unit_name in cards:
            cards.remove(unit_name)
            counts[unit_name] -= 1
            swap_count += 1

            # Add a spell or gear
            if spell_deficit > 0 and spell_idx < len(available_spells):
                replacement = available_spells[spell_idx]
                cards.append(replacement.name)
                counts[replacement.name] += 1
                if counts[replacement.name] >= replacement.max_copies:
                    spell_idx += 1
                spell_deficit -= 1
            elif gear_deficit > 0 and gear_idx < len(available_gear):
                replacement = available_gear[gear_idx]
                cards.append(replacement.name)
                counts[replacement.name] += 1
                if counts[replacement.name] >= replacement.max_copies:
                    gear_idx += 1
                gear_deficit -= 1
            elif spell_idx < len(available_spells):
                replacement = available_spells[spell_idx]
                cards.append(replacement.name)
                counts[replacement.name] += 1
                if counts[replacement.name] >= replacement.max_copies:
                    spell_idx += 1
            elif gear_idx < len(available_gear):
                replacement = available_gear[gear_idx]
                cards.append(replacement.name)
                counts[replacement.name] += 1
                if counts[replacement.name] >= replacement.max_copies:
                    gear_idx += 1

    return cards
