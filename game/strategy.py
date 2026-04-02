"""
Expert heuristic AI strategy for Riftbound simulation.

All decisions are DYNAMIC — based on the specific board state, matchup,
hand contents, and opponent resources. No hard-coded "always do X" rules.

The AI thinks in terms of:
  - What does my opponent's domain threaten? (removal, aggro, control)
  - What's my win condition and is it safe to deploy?
  - What protection do I have available?
  - Is the opponent tapped out (low energy/runes) = safe window?
  - Am I ahead or behind, and what does that change?
"""

import random


# ---------------------------------------------------------------------------
# Domain threat profiles — what each domain is known for
# ---------------------------------------------------------------------------

DOMAIN_THREATS = {
    "Order": {
        "removal_density": 0.8,
        "aggro_density":   0.3,
        "control_density": 0.7,
        "key_removal_cost": 3,
        "combat_trick_might": 2,   # moderate combat tricks
    },
    "Fury": {
        "removal_density": 0.5,
        "aggro_density":   0.9,
        "control_density": 0.2,
        "key_removal_cost": 2,
        "combat_trick_might": 3,   # damage-based tricks
    },
    "Chaos": {
        "removal_density": 0.6,
        "aggro_density":   0.6,
        "control_density": 0.5,
        "key_removal_cost": 2,
        "combat_trick_might": 3,
    },
    "Body": {
        "removal_density": 0.3,
        "aggro_density":   0.7,
        "control_density": 0.4,
        "key_removal_cost": 4,
        "combat_trick_might": 5,   # Punch First = +5 Might
    },
    "Calm": {
        "removal_density": 0.4,
        "aggro_density":   0.4,
        "control_density": 0.6,
        "key_removal_cost": 3,
        "combat_trick_might": 3,   # Defy, Discipline
    },
    "Mind": {
        "removal_density": 0.7,
        "aggro_density":   0.3,
        "control_density": 0.8,
        "key_removal_cost": 2,
        "combat_trick_might": 2,
    },
}


def _opponent_threat_profile(opponent):
    """Build a threat profile from the opponent's rune pool domains."""
    d1 = getattr(opponent.rune_pool, 'domain1', None)
    d2 = getattr(opponent.rune_pool, 'domain2', None)

    profile = {"removal_risk": 0.0, "aggro_risk": 0.0, "control_risk": 0.0,
               "min_removal_cost": 10, "combat_trick_might": 0}

    for domain in [d1, d2]:
        if domain and domain in DOMAIN_THREATS:
            t = DOMAIN_THREATS[domain]
            profile["removal_risk"] = max(profile["removal_risk"], t["removal_density"])
            profile["aggro_risk"]   = max(profile["aggro_risk"], t["aggro_density"])
            profile["control_risk"] = max(profile["control_risk"], t["control_density"])
            profile["min_removal_cost"] = min(profile["min_removal_cost"], t["key_removal_cost"])
            profile["combat_trick_might"] = max(profile["combat_trick_might"], t.get("combat_trick_might", 0))

    return profile


# ---------------------------------------------------------------------------
# Situational assessments
# ---------------------------------------------------------------------------

def _board_state(player, opponent, battlefields):
    """Assess the current board position."""
    friendly_might = 0
    enemy_might = 0
    friendly_count = 0
    enemy_count = 0

    for bf in battlefields:
        f_units = [u for u in bf.get_units(player.name) if u.is_alive]
        e_units = [u for u in bf.get_units(opponent.name) if u.is_alive]
        friendly_might += sum(u.effective_might for u in f_units)
        enemy_might += sum(u.effective_might for u in e_units)
        friendly_count += len(f_units)
        enemy_count += len(e_units)

    return {
        "friendly_might": friendly_might,
        "enemy_might": enemy_might,
        "friendly_count": friendly_count,
        "enemy_count": enemy_count,
        "behind": enemy_might > friendly_might * 1.2,
        "ahead": friendly_might > enemy_might * 1.2,
        "score_lead": player.score - opponent.score,
    }


def _has_protection_in_hand(player):
    """Check if hand contains protection pieces (gear/spells that protect units)."""
    protection_keywords = ["guardian angel", "zhonya", "shield", "unyielding", "not so fast",
                          "riposte", "defy", "wind wall", "counter"]
    for card in player.hand:
        ability = card.ability.lower()
        name = card.name.lower()
        for kw in protection_keywords:
            if kw in ability or kw in name:
                return True
    return False


def _has_protection_on_board(player, battlefields):
    """Check if we have protective gear/units already in play."""
    for bf in battlefields:
        for u in bf.get_units(player.name):
            if u.is_alive and (u.has("Shield") or u.has("Tank")):
                return True
    return False


def _opponent_can_remove(opponent, cost_threshold=0):
    """Check if opponent has enough resources to cast removal."""
    return opponent.energy >= cost_threshold and opponent.rune_pool.pool >= 1


def _opponent_rune_pressure(opponent):
    """
    Assess how much rune pressure the opponent is under.
    Low runes early = they're crippled for tempo.
    Returns 0.0 (flush with runes) to 1.0 (completely starved).
    """
    total_runes = opponent.rune_pool.pool
    total_channeled = opponent.rune_pool.total_channeled
    max_possible = 12  # max rune slots

    if total_channeled == 0:
        return 0.0

    # How much of their channeled runes have they spent?
    spent = total_channeled - total_runes
    return min(1.0, spent / max(total_channeled, 1))


def _removal_cost_against(card, opponent):
    """
    Estimate how many resources the opponent would need to remove this card.
    Accounts for Deflect (forces extra rune payment per targeting instance).
    """
    threat = _opponent_threat_profile(opponent)
    base_cost = threat["min_removal_cost"]  # energy cost of cheapest removal

    # Deflect: opponent must pay extra runes for each instance that targets this unit
    deflect_val = card.keyword_value("Deflect", 0)
    if deflect_val > 0:
        # Most removal targets once (1 extra rune), but multi-hit spells
        # like Falling Star hit twice = 2 extra runes.
        # Average removal costs 1-2 targeting instances
        base_cost += deflect_val * 1.5  # conservative estimate of extra rune tax

    return base_cost


def _has_follow_up(player, max_cost=4):
    """
    Check if the player has a strong follow-up play for next turn.
    If yes, a 'bait' play this turn is less risky because you have a plan B.
    """
    follow_ups = [c for c in player.hand if c.cost <= max_cost and c.card_type == "Unit"]
    return len(follow_ups) >= 1


def _is_win_condition(card, hand):
    """
    Determine if a card is a key piece the deck revolves around.
    Win conditions: the card you MUST protect, not throw away carelessly.
    """
    if not card.champion:
        return False

    copies_in_hand = sum(1 for c in hand if c.name == card.name)
    return copies_in_hand <= 1  # last copy = win condition


# ---------------------------------------------------------------------------
# Dynamic card scoring
# ---------------------------------------------------------------------------

def card_play_score(card, player, opponent, battlefields):
    """
    Score how good it is to play this card RIGHT NOW.

    This is fully dynamic — the same card gets different scores depending on:
      - Board state (ahead/behind/even)
      - Opponent's domain (removal risk)
      - What else is in your hand (protection pieces)
      - Opponent's resources (tapped out = safe window)
      - Whether this card is your win condition
    """
    board = _board_state(player, opponent, battlefields)
    threat = _opponent_threat_profile(opponent)
    has_protection = _has_protection_in_hand(player)
    has_board_protection = _has_protection_on_board(player, battlefields)
    opp_can_remove = _opponent_can_remove(opponent, threat["min_removal_cost"])
    opp_rune_pressure = _opponent_rune_pressure(opponent)
    has_followup = _has_follow_up(player)

    score = 0.0

    if card.card_type == "Unit":
        efficiency = card.health / max(card.cost, 1)
        score = efficiency * 3

        if _is_win_condition(card, player.hand):
            # --- WIN CONDITION LOGIC ---
            # Dynamic decision: play, hold, or BAIT based on full situation

            removal_cost = _removal_cost_against(card, opponent)
            has_deflect = card.has("Deflect")

            if has_deflect:
                # --- DEFLECT CHAMPIONS: BAIT POTENTIAL ---
                # Deflect makes removal expensive. If the opponent tries to
                # remove this, they burn extra runes per targeting instance.
                # Playing this "unprotected" can be INTENTIONAL bait.

                if opp_can_remove and removal_cost > opponent.rune_pool.pool * 0.5:
                    # Removing this would cost the opponent a huge chunk of runes
                    # It's a GOOD bait — they either leave it alive (we win)
                    # or they burn runes trying to kill it (we win on tempo)
                    if has_followup:
                        score += 7   # strong bait: we have a follow-up play
                    else:
                        score += 4   # decent bait but no backup plan

                elif not opp_can_remove:
                    score += 8   # tapped out = free champion

                elif removal_cost <= 2:
                    # Cheap removal exists even through Deflect — be careful
                    if has_protection:
                        score += 3
                    else:
                        score -= 3   # hold it

                else:
                    score += 5   # Deflect provides natural protection

            elif opp_can_remove and threat["removal_risk"] > 0.5:
                # --- NON-DEFLECT CHAMPION vs REMOVAL DOMAIN ---
                if has_protection or has_board_protection:
                    score += 2   # play it, but let protection land first
                elif opponent.energy <= 2:
                    score += 3   # narrow window
                elif opp_rune_pressure > 0.6:
                    # Opponent already spent most of their runes this game
                    # They may not have enough for removal even if they have the card
                    score += 4
                else:
                    score -= 8   # HOLD — too risky
            elif not opp_can_remove:
                score += 8       # tapped out = slam it
            else:
                score += 5       # low removal risk domain

        else:
            # --- REGULAR UNIT ---
            if board["behind"]:
                score *= 1.5   # need board presence
            if card.champion:
                score += 3     # champions are impactful but not the key piece

            # Prefer playing cheap units first to bait removal before key pieces
            if card.cost <= 3:
                score += 1     # cheap units are good removal bait

        # Keyword situational value
        score += card.keyword_value("Assault") * (2.0 if board["behind"] else 1.0)
        score += card.keyword_value("Shield") * (2.0 if board["ahead"] else 1.0)
        if card.has("Tank"):
            # Tanks are great when you have valuable units to protect
            has_valuable = any(
                u.card.champion for bf in battlefields
                for u in bf.get_units(player.name) if u.is_alive
            )
            score += 4 if has_valuable else 1
        if card.has("Ganking"):
            score += 2

    elif card.card_type == "Spell":
        ability = card.ability.lower()

        if "deal" in ability or "kill" in ability or "destroy" in ability:
            # --- REMOVAL ---
            # Value depends on whether there's a high-value target to remove
            best_target_value = 0
            for bf in battlefields:
                for u in bf.get_units(opponent.name):
                    if u.is_alive:
                        val = u.effective_might
                        if u.card.champion:
                            val += 10  # removing their champion is huge
                        best_target_value = max(best_target_value, val)

            if best_target_value > 0:
                score = 3 + best_target_value * 0.5
                if best_target_value >= 8:
                    score += 3   # high-value removal target = use it now
            else:
                score = 0.5   # no targets — hold it for later

        elif "draw" in ability:
            hand_size = len(player.hand)
            score = 5 if hand_size <= 2 else 2

        elif "ready" in ability:
            exhausted = sum(1 for bf in battlefields
                          for u in bf.get_units(player.name) if u.is_exhausted)
            score = 2 + exhausted * 2

        elif "counter" in ability or "reaction" in ability:
            # Hold reactive spells — don't play them proactively
            score = 0.5

        elif any(kw in ability for kw in ["guardian angel", "zhonya", "shield", "protect"]):
            # --- PROTECTION ---
            # Valuable when we have a champion on board to protect
            champs_on_board = sum(
                1 for bf in battlefields
                for u in bf.get_units(player.name)
                if u.is_alive and u.card.champion
            )
            if champs_on_board > 0:
                score = 7   # protect our champion NOW
            else:
                score = 2   # hold for when we deploy the champion
        else:
            score = 3

    elif card.card_type == "Gear":
        # Gear is best when we have units, especially champions
        champs_on_board = sum(
            1 for bf in battlefields
            for u in bf.get_units(player.name)
            if u.is_alive and u.card.champion
        )
        has_units = any(bf.get_units(player.name) for bf in battlefields)

        ability = card.ability.lower()
        is_protective = any(kw in ability or kw in card.name.lower()
                          for kw in ["guardian angel", "zhonya", "shield", "armor"])

        if is_protective and champs_on_board > 0:
            score = 8   # protect the champion!
        elif is_protective:
            # Hold protective gear for when champion comes down
            has_champ_in_hand = any(c.champion for c in player.hand)
            score = 2 if has_champ_in_hand else 4
        elif has_units:
            score = 3
        else:
            score = 1

    return score


# ---------------------------------------------------------------------------
# Board evaluation helpers
# ---------------------------------------------------------------------------

def battlefield_value(bf, player_name, opponent_name):
    """Score a battlefield from the player's perspective."""
    friendly = bf.get_units(player_name)
    enemy    = bf.get_units(opponent_name)

    f_might = sum(u.effective_might for u in friendly if u.is_alive)
    e_might = sum(u.effective_might for u in enemy if u.is_alive)

    score = bf.point_value * 3

    if bf.controller and bf.controller.name == player_name:
        score += 5
    elif bf.controller and bf.controller.name == opponent_name:
        score += 8

    if e_might > 0:
        score += 3
        if f_might > e_might:
            score += 2
        else:
            score += 4

    if not enemy and bf.controller is None:
        score += 2

    return score


def unit_threat(unit):
    """Rate how threatening an enemy unit is."""
    score = unit.effective_might * 2
    if unit.card.champion:
        score += 10
    if unit.has("Assault"):
        score += unit.keyword_value("Assault") * 2
    if unit.has("Tank"):
        score += 3
    if unit.has("Ganking"):
        score += 4
    return score


def unit_value(unit):
    """Rate how valuable a friendly unit is."""
    score = unit.effective_might * 2
    if unit.card.champion:
        score += 8
    if unit.has("Assault"):
        score += unit.keyword_value("Assault") * 2
    if not unit.is_exhausted:
        score += 3
    return score


# ---------------------------------------------------------------------------
# Expert strategy class
# ---------------------------------------------------------------------------

class ExpertStrategy:
    """
    Dynamic decision-making that adapts to every game situation.
    No hard rules — every decision is a weighted score based on
    board state, matchup, hand contents, and opponent resources.
    """

    # --- Card selection ---

    def choose_cards_to_play(self, player, battlefields, opponent):
        """
        Decide which cards to play and in what order.
        Adapts to matchup, board state, and opponent resources.
        """
        affordable = [c for c in player.hand if player.can_afford(c)]
        if not affordable:
            return []

        scored = [(card_play_score(c, player, opponent, battlefields), c) for c in affordable]
        scored.sort(key=lambda x: x[0], reverse=True)

        to_play = []
        sim_energy = player.energy
        sim_runes  = player.rune_pool.pool

        for score, card in scored:
            if sim_energy >= card.cost and sim_runes >= card.rune_cost:
                # Skip cards with negative or very low scores (AI decided to hold them)
                if score <= 0:
                    continue

                remaining_energy = sim_energy - card.cost

                # Resource holdback: if we have a high-value card we're saving,
                # don't spend all resources on low-value plays
                unplayed_better = [
                    (s, c) for s, c in scored
                    if c not in to_play and c is not card
                    and c.cost > remaining_energy
                    and s > score * 1.3 and s > 0
                ]
                real_holdbacks = [(s, c) for s, c in unplayed_better if c.cost <= player.max_energy + 1]
                if real_holdbacks and score < 4:
                    continue

                to_play.append(card)
                sim_energy -= card.cost
                sim_runes  -= card.rune_cost

        return to_play

    # --- Deployment ---

    def choose_battlefield(self, player, opponent_name, battlefields):
        """Pick the best battlefield based on scoring, control, and might balance."""
        if not battlefields:
            return None

        scored = []
        for bf in battlefields:
            score = battlefield_value(bf, player.name, opponent_name)
            friendly = bf.get_units(player.name)
            if len(friendly) >= 4:
                score *= 0.5
            elif len(friendly) >= 6:
                score *= 0.2
            scored.append((score, bf))

        scored.sort(key=lambda x: x[0], reverse=True)

        if len(scored) >= 2 and scored[0][0] - scored[1][0] < 2:
            top = scored[:2]
            return random.choice(top)[1]

        return scored[0][1]

    # --- Combat ---

    def should_attack(self, bf, attacker, defender):
        """
        Decide whether to attack — considers might, keywords,
        champion risk, AND potential combat tricks from untapped runes.
        """
        atk_units = [u for u in bf.get_units(attacker.name) if not u.is_exhausted]
        def_units = bf.get_units(defender.name)

        if not atk_units:
            return False
        if not def_units:
            return True

        atk_might = sum(u.effective_might for u in atk_units)
        def_might = sum(u.effective_might for u in def_units)

        for u in atk_units:
            atk_might += u.keyword_value("Assault")
        for u in def_units:
            def_might += u.keyword_value("Shield")

        for u in atk_units:
            if u.has("Stun"):
                atk_might -= u.effective_might
        for u in def_units:
            if u.has("Stun"):
                def_might -= u.effective_might

        # --- Combat trick awareness ---
        # If the defender has untapped runes, they might have a combat trick.
        # Body with 1 rune = Punch First (+5 Might) is a real threat.
        trick_bonus = 0
        if defender.rune_pool.pool > 0 and defender.energy > 0:
            threat = _opponent_threat_profile(defender)
            trick_bonus = threat.get("combat_trick_might", 0)

        # Factor trick into effective defense — assume they might have it
        effective_def = def_might + trick_bonus

        we_control = bf.controller and bf.controller.name == attacker.name
        they_control = bf.controller and bf.controller.name == defender.name

        # Would we lose a champion in this attack (including possible trick)?
        champs_at_risk = [u for u in atk_units if u.card.champion]
        if champs_at_risk:
            # If trick could flip the trade and kill our champion, don't attack
            if effective_def >= atk_might * 0.8:
                if not they_control:
                    return False  # not worth risking champion

        # Use effective_def (with trick potential) for close calls
        if atk_might >= effective_def * 1.3:
            return True   # overwhelming advantage even with trick
        if atk_might >= def_might * 1.3 and trick_bonus > 0:
            # We beat their base might but a trick could swing it
            # Only attack if the payoff is worth the risk
            if they_control:
                return True   # must contest even with trick risk
            return False      # not worth it for a neutral bf

        if atk_might >= effective_def * 0.9:
            if they_control:
                return True
            if not we_control:
                return True
            return False
        if they_control and atk_might >= effective_def * 0.7:
            return True

        return False

    # --- Spell targeting ---

    def pick_damage_target(self, opponent, battlefields, bf_hint=None):
        """Target the highest-threat enemy, preferring killable targets."""
        candidates = []
        if bf_hint:
            candidates = bf_hint.get_units(opponent.name)
        if not candidates:
            for bf in battlefields:
                candidates.extend(bf.get_units(opponent.name))

        alive = [u for u in candidates if u.is_alive]
        if not alive:
            return None

        def target_priority(u):
            threat = unit_threat(u)
            if u.current_health <= 3:
                threat += 10  # killable = bonus
            return threat

        return max(alive, key=target_priority)

    def pick_buff_target(self, caster, battlefields):
        """Buff the unit that benefits most — ready units and champions first."""
        candidates = []
        for bf in battlefields:
            candidates.extend(bf.get_units(caster.name))
        alive = [u for u in candidates if u.is_alive]
        if not alive:
            return None

        def buff_priority(u):
            val = unit_value(u)
            if not u.is_exhausted:
                val += 5
            return val

        return max(alive, key=buff_priority)

    def pick_bounce_target(self, opponent, battlefields):
        """Bounce the most expensive/threatening enemy unit."""
        candidates = []
        for bf in battlefields:
            candidates.extend(bf.get_units(opponent.name))
        alive = [u for u in candidates if u.is_alive]
        if not alive:
            return None
        return max(alive, key=lambda u: u.card.cost + unit_threat(u) * 0.5)

    def pick_ready_target(self, caster, battlefields):
        """Ready the highest-might exhausted unit."""
        candidates = []
        for bf in battlefields:
            candidates.extend(bf.get_units(caster.name))
        candidates.extend(caster.base_units)
        exhausted = [u for u in candidates if u.is_exhausted and u.is_alive]
        if not exhausted:
            return None
        return max(exhausted, key=lambda u: u.effective_might)
