"""
Expert heuristic AI strategy for Riftbound simulation.

Replaces every in-game decision point with logic that mimics
how a top-level player would think:

1. Card selection:  evaluate board state, play for tempo or value
2. Deployment:      put units where they win fights and score points
3. Combat:          only attack when the trade is favorable
4. Spell targeting: remove biggest threats, buff best attackers
5. Resource mgmt:   hold mana if next turn has a better line
"""

import random


# ---------------------------------------------------------------------------
# Board evaluation helpers
# ---------------------------------------------------------------------------

def battlefield_value(bf, player_name, opponent_name):
    """
    Score a battlefield from the player's perspective.
    Higher = more important to contest.
    """
    friendly = bf.get_units(player_name)
    enemy    = bf.get_units(opponent_name)

    f_might = sum(u.effective_might for u in friendly if u.is_alive)
    e_might = sum(u.effective_might for u in enemy if u.is_alive)

    score = bf.point_value * 3     # base value by points

    # Battlefields we control are worth defending
    if bf.controller and bf.controller.name == player_name:
        score += 5    # hold bonus — we score every turn
    # Battlefields they control are worth contesting
    elif bf.controller and bf.controller.name == opponent_name:
        score += 8    # conquer bonus — deny their points and take one

    # Might advantage: we want to deploy where we can win
    if e_might > 0:
        score += 3    # contested = important
        if f_might > e_might:
            score += 2    # we're winning here — reinforce
        else:
            score += 4    # we're losing — need help

    # Empty uncontrolled = free points if we get there
    if not enemy and bf.controller is None:
        score += 2

    return score


def unit_threat(unit):
    """
    Rate how threatening an enemy unit is. Higher = should be removed first.
    Champions and high-might units are top priority.
    """
    score = unit.effective_might * 2

    if unit.card.champion:
        score += 10    # champions are high-value targets

    if unit.has("Assault"):
        score += unit.keyword_value("Assault") * 2
    if unit.has("Tank"):
        score += 3     # tanks protect other units
    if unit.has("Ganking"):
        score += 4     # mobile units are dangerous

    return score


def unit_value(unit):
    """
    Rate how valuable a friendly unit is. Higher = protect and buff first.
    """
    score = unit.effective_might * 2

    if unit.card.champion:
        score += 8
    if unit.has("Assault"):
        score += unit.keyword_value("Assault") * 2
    if not unit.is_exhausted:
        score += 3     # ready units are more valuable (can attack)

    return score


def card_play_score(card, player, opponent, battlefields):
    """
    Score a card for how good it is to play RIGHT NOW given the board state.
    Higher = play this first.
    """
    score = 0.0

    # --- Board state assessment ---
    friendly_total = 0
    enemy_total = 0
    for bf in battlefields:
        friendly_total += sum(u.effective_might for u in bf.get_units(player.name) if u.is_alive)
        enemy_total    += sum(u.effective_might for u in bf.get_units(opponent.name) if u.is_alive)

    behind = enemy_total > friendly_total * 1.2
    ahead  = friendly_total > enemy_total * 1.2

    if card.card_type == "Unit":
        # Units are highest priority when behind (need board presence)
        efficiency = card.health / max(card.cost, 1)
        score = efficiency * 3

        if behind:
            score *= 1.5   # premium on board presence when losing
        if card.champion:
            score += 4     # champions are high impact

        # Keyword bonuses
        score += card.keyword_value("Assault") * 1.5
        score += card.keyword_value("Shield") * 1.0
        if card.has("Tank"):
            score += 2 if behind else 0.5
        if card.has("Ganking"):
            score += 2

    elif card.card_type == "Spell":
        # Removal spells are better when opponent has high-value targets
        ability = card.ability.lower()

        if "deal" in ability or "kill" in ability or "destroy" in ability:
            # Removal — valuable when enemy has threats on board
            if enemy_total > 0:
                score = 5 + (2 if not behind else 4)
            else:
                score = 1    # no targets = low priority
        elif "draw" in ability:
            # Card draw — good when hand is small
            hand_size = len(player.hand)
            score = 4 if hand_size <= 2 else 2
        elif "ready" in ability:
            # Ready effects — good when we have exhausted units
            exhausted = sum(1 for bf in battlefields
                          for u in bf.get_units(player.name) if u.is_exhausted)
            score = 3 + exhausted
        else:
            # Other spells
            score = 3

    elif card.card_type == "Gear":
        # Gear — good when we have units to equip
        has_units = any(bf.get_units(player.name) for bf in battlefields)
        score = 3 if has_units else 1

    # Cost efficiency penalty — very expensive cards are riskier
    if card.cost >= 7:
        score -= 1

    return score


# ---------------------------------------------------------------------------
# Expert strategy class
# ---------------------------------------------------------------------------

class ExpertStrategy:
    """
    Pluggable decision-making for a Player.
    Each method replaces a decision point in player.py or engine.py.
    """

    # --- Card selection ---

    def choose_cards_to_play(self, player, battlefields, opponent):
        """
        Decide which cards to play and in what order.
        Returns list of cards to play (in order).

        Key differences from basic AI:
        - Evaluates board state to prioritize units vs spells
        - Considers saving resources for next turn
        - Doesn't blindly play everything
        """
        affordable = [c for c in player.hand if player.can_afford(c)]
        if not affordable:
            return []

        # Score each card by how good it is to play right now
        scored = [(card_play_score(c, player, opponent, battlefields), c) for c in affordable]
        scored.sort(key=lambda x: x[0], reverse=True)

        to_play = []
        sim_energy = player.energy
        sim_runes  = player.rune_pool.pool

        for score, card in scored:
            if sim_energy >= card.cost and sim_runes >= card.rune_cost:
                # Resource hold-back: if we have a high-value card we can't
                # afford yet but could next turn, don't spend everything
                remaining_energy = sim_energy - card.cost
                remaining_runes  = sim_runes - card.rune_cost

                # Check if there's a better card in hand we're saving for
                unplayed_better = [
                    c for _, c in scored
                    if c not in to_play and c is not card
                    and c.cost > remaining_energy
                    and card_play_score(c, player, opponent, battlefields) > score * 1.3
                ]

                # If the only unplayed "better" cards cost more than we'd ever
                # have (cost > 10), don't hold back for them
                real_holdbacks = [c for c in unplayed_better if c.cost <= player.max_energy + 1]

                if real_holdbacks and score < 4:
                    # Skip this low-value play to save resources
                    continue

                to_play.append(card)
                sim_energy -= card.cost
                sim_runes  -= card.rune_cost

        return to_play

    # --- Deployment ---

    def choose_battlefield(self, player, opponent_name, battlefields):
        """
        Pick the best battlefield to deploy a unit to.

        Key differences from basic AI:
        - Evaluates point value, control state, and might balance
        - Focuses on battlefields that score points
        - Avoids over-committing to already-won lanes
        """
        if not battlefields:
            return None

        scored = []
        for bf in battlefields:
            score = battlefield_value(bf, player.name, opponent_name)

            # Diminishing returns: don't stack 5+ units at one bf
            friendly = bf.get_units(player.name)
            if len(friendly) >= 4:
                score *= 0.5
            elif len(friendly) >= 6:
                score *= 0.2

            scored.append((score, bf))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Small randomness to avoid being too predictable
        if len(scored) >= 2 and scored[0][0] - scored[1][0] < 2:
            top = scored[:2]
            return random.choice(top)[1]

        return scored[0][1]

    # --- Combat ---

    def should_attack(self, bf, attacker, defender):
        """
        Decide whether to attack at this battlefield.
        Returns True if we should initiate combat.

        Key differences from basic AI:
        - Evaluates whether we win the trade
        - Considers holding position for next turn's score
        - Factors in keyword advantages
        """
        atk_units = [u for u in bf.get_units(attacker.name) if not u.is_exhausted]
        def_units = bf.get_units(defender.name)

        if not atk_units:
            return False

        if not def_units:
            # No defenders — free conquer, always attack
            return True

        # Calculate effective combat power
        atk_might = sum(u.effective_might for u in atk_units)
        def_might = sum(u.effective_might for u in def_units)

        # Keyword adjustments
        for u in atk_units:
            atk_might += u.keyword_value("Assault")
        for u in def_units:
            def_might += u.keyword_value("Shield")

        # Stun check: stunned units contribute 0
        for u in atk_units:
            if u.has("Stun"):
                atk_might -= u.effective_might
        for u in def_units:
            if u.has("Stun"):
                def_might -= u.effective_might

        # Decision logic:
        # - If we have clear advantage (>= 1.3x their might), attack
        # - If it's close but we control the bf, hold (we score passively)
        # - If they control and we're close, attack (deny their score)
        # - If we're clearly weaker, don't throw units away

        we_control = bf.controller and bf.controller.name == attacker.name
        they_control = bf.controller and bf.controller.name == defender.name

        if atk_might >= def_might * 1.3:
            return True    # clear advantage

        if atk_might >= def_might * 0.9:
            if they_control:
                return True    # close fight but we need to contest
            if not we_control:
                return True    # neutral bf, worth contesting
            # We control and it's close — hold, score passively
            return False

        if they_control and atk_might >= def_might * 0.7:
            return True    # desperate contest — can't let them hold forever

        return False    # too weak, don't suicide

    # --- Spell targeting ---

    def pick_damage_target(self, opponent, battlefields, bf_hint=None):
        """
        Pick the best enemy unit to deal damage to.

        Key differences from basic AI:
        - Targets highest-threat unit, not weakest
        - Prefers units we can actually kill (lethal range)
        - Champions are priority targets
        """
        candidates = []
        if bf_hint:
            candidates = bf_hint.get_units(opponent.name)
        if not candidates:
            for bf in battlefields:
                candidates.extend(bf.get_units(opponent.name))

        alive = [u for u in candidates if u.is_alive]
        if not alive:
            return None

        # Priority: killable targets first (finish them off)
        # then highest threat
        def target_priority(u):
            threat = unit_threat(u)
            # Bonus for targets in lethal range (health <= 3)
            if u.current_health <= 3:
                threat += 10
            return threat

        return max(alive, key=target_priority)

    def pick_buff_target(self, caster, battlefields):
        """
        Pick the best friendly unit to buff.
        Prefers ready units (can attack this turn) and high-value units.
        """
        candidates = []
        for bf in battlefields:
            candidates.extend(bf.get_units(caster.name))

        alive = [u for u in candidates if u.is_alive]
        if not alive:
            return None

        def buff_priority(u):
            val = unit_value(u)
            if not u.is_exhausted:
                val += 5    # ready = can use the buff immediately
            return val

        return max(alive, key=buff_priority)

    def pick_bounce_target(self, opponent, battlefields):
        """
        Pick the best enemy unit to bounce (return to hand).
        Targets the highest-cost unit (wastes the most of their tempo).
        """
        candidates = []
        for bf in battlefields:
            candidates.extend(bf.get_units(opponent.name))

        alive = [u for u in candidates if u.is_alive]
        if not alive:
            return None

        # Bounce the most expensive unit (maximum tempo loss)
        return max(alive, key=lambda u: u.card.cost + unit_threat(u) * 0.5)

    def pick_ready_target(self, caster, battlefields):
        """Pick the best exhausted friendly unit to ready."""
        candidates = []
        for bf in battlefields:
            candidates.extend(bf.get_units(caster.name))
        candidates.extend(caster.base_units)

        exhausted = [u for u in candidates if u.is_exhausted and u.is_alive]
        if not exhausted:
            return None

        # Ready the highest-might exhausted unit
        return max(exhausted, key=lambda u: u.effective_might)
