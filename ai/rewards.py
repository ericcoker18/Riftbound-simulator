"""
Turn-level reward shaping for Riftbound RL training.

Instead of only learning from win/loss at game end, the agent gets
intermediate rewards every turn that reflect how a pro evaluates
board position. This dramatically speeds up learning.

A pro player thinks: "That turn was good because..."
  - I gained board control (+)
  - I traded up (killed a champion with a cheap unit) (+)
  - I held my win condition safely (+)
  - I wasted removal on a weak target (-)
  - I overextended and lost my champion (-)
  - I'm closer to scoring (+ / -)
"""


def evaluate_position(player, opponent, battlefields):
    """
    Score the current board position from player's perspective.
    Returns a float: positive = ahead, negative = behind.

    This captures everything a pro evaluates at a glance:
      - Score lead
      - Board presence (total might)
      - Battlefield control
      - Card advantage (hand size)
      - Champion status
      - Resource advantage
    """
    score = 0.0

    # --- Score lead (most important) ---
    score_diff = player.score - opponent.score
    score += score_diff * 3.0  # each point is worth a lot

    # --- Board presence ---
    friendly_might = 0
    enemy_might = 0
    friendly_champs = 0
    enemy_champs = 0
    bfs_controlled = 0
    bfs_enemy = 0

    for bf in battlefields:
        f_units = [u for u in bf.get_units(player.name) if u.is_alive]
        e_units = [u for u in bf.get_units(opponent.name) if u.is_alive]

        friendly_might += sum(u.effective_might for u in f_units)
        enemy_might += sum(u.effective_might for u in e_units)

        friendly_champs += sum(1 for u in f_units if u.card.champion)
        enemy_champs += sum(1 for u in e_units if u.card.champion)

        if bf.controller and bf.controller.name == player.name:
            bfs_controlled += 1
        elif bf.controller and bf.controller.name == opponent.name:
            bfs_enemy += 1

    # Might advantage
    might_diff = friendly_might - enemy_might
    score += might_diff * 0.3

    # Battlefield control
    score += (bfs_controlled - bfs_enemy) * 2.0

    # Champion advantage — having your champion alive is huge
    score += (friendly_champs - enemy_champs) * 1.5

    # --- Card advantage ---
    hand_diff = len(player.hand) - len(opponent.hand)
    score += hand_diff * 0.5

    # Deck remaining (running out is bad)
    deck_diff = len(player.deck.cards) - len(opponent.deck.cards)
    score += deck_diff * 0.1

    # --- Resource advantage ---
    energy_diff = player.energy - opponent.energy
    rune_diff = player.rune_pool.pool - opponent.rune_pool.pool
    score += energy_diff * 0.1
    score += rune_diff * 0.2

    # --- Spell/gear usage reward ---
    # Encourage the AI to actually use spells and gear, not just units
    history = getattr(player, '_game_history', None)
    if history:
        spells_played = sum(1 for c in history.cards_played.get(player.name, [])
                           if c.card_type == "Spell")
        gear_played = sum(1 for c in history.cards_played.get(player.name, [])
                         if c.card_type == "Gear")
        # Bonus for spell/gear usage (diminishing returns)
        score += min(spells_played, 5) * 0.3
        score += min(gear_played, 3) * 0.4

    return score


def compute_turn_reward(prev_position, curr_position, game_over=False, won=False):
    """
    Compute reward for a single turn based on position change.

    reward = (how much position improved this turn) + (game outcome bonus)

    This teaches the agent:
      - Turns that improve position = good
      - Turns that worsen position = bad
      - Winning = big bonus, losing = big penalty
    """
    # Position delta — did this turn make things better or worse?
    delta = curr_position - prev_position

    # Scale to reasonable range
    turn_reward = delta * 0.1

    # Game outcome bonus (dominates but doesn't erase turn signal)
    if game_over:
        turn_reward += 5.0 if won else -5.0

    # Clamp to prevent extreme values
    return max(-3.0, min(3.0, turn_reward))


# ---------------------------------------------------------------------------
# Detailed turn analysis (for learning specific patterns)
# ---------------------------------------------------------------------------

def analyze_turn(player, opponent, battlefields, history=None):
    """
    Detailed turn analysis that identifies specific good/bad decisions.
    Returns a dict of named reward components for debugging.
    """
    analysis = {}

    # Did we trade up? (killed expensive units with cheap ones)
    if history:
        opp_dead_value = 0
        our_dead_value = 0
        for c in history.cards_played.get(opponent.name, [])[-3:]:
            if c.card_type == "Unit":
                opp_dead_value += c.cost
        for c in history.cards_played.get(player.name, [])[-3:]:
            if c.card_type == "Unit":
                our_dead_value += c.cost

    # Champion safety
    our_champs_alive = sum(
        1 for bf in battlefields
        for u in bf.get_units(player.name)
        if u.is_alive and u.card.champion
    )
    opp_champs_alive = sum(
        1 for bf in battlefields
        for u in bf.get_units(opponent.name)
        if u.is_alive and u.card.champion
    )
    analysis["champion_advantage"] = our_champs_alive - opp_champs_alive

    # Tempo: did we spend our resources efficiently?
    energy_used = player.max_energy - player.energy
    analysis["energy_efficiency"] = energy_used / max(player.max_energy, 1)

    # Did we waste removal on a weak target?
    if history:
        recent_removal = [
            c for c in history.removal_used.get(player.name, [])[-2:]
        ]
        analysis["removal_used_this_turn"] = len(recent_removal)

    return analysis
