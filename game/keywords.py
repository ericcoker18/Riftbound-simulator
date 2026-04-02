"""
Keyword definitions sourced from Riftbound Core Rules (2025-12-01).
Ambush and Hunt are not in the core rulebook — definitions extracted from card text.

SIMULATABLE (implemented in engine.py):
  Assault N  [733] — While attacking, I have +N Might.
  Shield N   [740] — While defending, I have +N Might.
  Tank       [741] — I must be assigned lethal damage before non-Tank units.
  Backline          — I must be assigned combat damage last. (from card text)
  Stun       [410] — A Stunned unit contributes 0 Might in combat.
                     It still requires lethal damage to be killed.
  Temporary  [742] — Killed at the start of my controller's Beginning Phase.
  Deathknell [734] — When I die, trigger [Effect]. (card-specific; logged only)
  Mighty     [706] — A unit is Mighty while its Might is 5 or greater.
                     Checked by other abilities.

NOT SIMULATABLE (too complex for current single-battlefield engine):
  Deflect N  [735] — Spells/abilities targeting me cost N extra Power.
  Ganking    [736] — I may move between battlefields.
  Hidden     [737] — May be hidden facedown; gains Reaction next turn.
  Accelerate [731] — Pay extra to enter ready instead of exhausted.
  Vision     [743] — When played, look at top deck card, may recycle.
  Legion     [738] — Bonus effect if another card was played this turn.
  Quick-Draw [745] — Gear: play and attach with Reaction timing.
  Weaponmaster[747]— When played, attach an Equipment at a discount.
  Ambush            — Two passives: (1) may be played to a battlefield where
                     you control units; (2) has Reaction when played there.
                     (Unleashed patch notes)
  Hunt N            — When this unit conquers or holds, gain N XP.
                     (Unleashed patch notes; XP system not yet simulated)
  Reaction   [739] — Can be played during Closed States.
  Action     [732] — Can be played during Showdowns.
  Repeat     [746] — Pay extra to execute the spell a second time.
"""

import re

# Keywords that carry a numeric value e.g. [Assault 2], [Shield 3]
VALUED_KEYWORDS = {"Assault", "Shield", "Deflect", "Hunt"}

# Keywords that are simple flags (present or not)
FLAG_KEYWORDS = {
    "Tank", "Backline", "Temporary", "Deathknell", "Mighty",
    "Stun", "Ganking", "Ambush", "Hidden", "Accelerate",
    "Vision", "Weaponmaster", "Legion", "Quick-Draw",
    "Predict", "Repeat", "Unique",
}


def parse_keywords(ability_text: str) -> dict:
    """
    Parse bracketed keywords from ability text into a dict.

    Valued keywords:  {"Assault": 2, "Shield": 1}
    Flag keywords:    {"Tank": True, "Backline": True}

    Unknown bracketed tokens are ignored.
    """
    keywords = {}
    if not ability_text:
        return keywords

    for match in re.finditer(r'\[([^\]]+)\]', ability_text):
        token = match.group(1).strip()

        parts = token.split()
        if len(parts) == 2 and parts[0] in VALUED_KEYWORDS:
            name, value = parts[0], int(parts[1])
            # Sum values if keyword appears more than once (e.g. Assault stacking)
            keywords[name] = keywords.get(name, 0) + value

        elif token in VALUED_KEYWORDS:
            # Valued keyword with implicit value of 1
            keywords[token] = keywords.get(token, 0) + 1

        elif token in FLAG_KEYWORDS:
            keywords[token] = True

    return keywords
