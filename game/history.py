"""
Game history and knowledge tracker.

Tracks everything a pro player would remember during a game:
  - Every card played by both players
  - Spells/removal used (and how many are likely left)
  - Cards revealed from hand
  - Behavior signals (passed with resources up)
  - Rune spending patterns
  - What the opponent's domain could theoretically have

This feeds into both the ExpertStrategy and the RL state encoder,
giving the AI the information it needs to make pro-level reads.
"""

from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# Known card pools per domain — what removal/tricks exist in each domain
# ---------------------------------------------------------------------------

# Key spells a pro would track. Format: {name: {"cost": E, "rune": R, "type": str}}
# type: "removal", "trick", "draw", "protection", "bounce"

DOMAIN_KEY_SPELLS = {
    "Order": {
        "Vengeance":        {"cost": 6, "rune": 1, "type": "removal"},
        "Cull the Weak":    {"cost": 3, "rune": 1, "type": "removal"},
        "Imperial Decree":  {"cost": 4, "rune": 1, "type": "removal"},
        "Divine Judgment":  {"cost": 5, "rune": 1, "type": "protection"},
        "Rally the Troops": {"cost": 3, "rune": 1, "type": "trick"},
        "Salvage":          {"cost": 3, "rune": 0, "type": "draw"},
    },
    "Fury": {
        "Falling Star":     {"cost": 2, "rune": 1, "type": "removal"},
        "Cleave":           {"cost": 3, "rune": 1, "type": "removal"},
        "Blood Rush":       {"cost": 1, "rune": 0, "type": "trick"},
        "Hextech Ray":      {"cost": 2, "rune": 1, "type": "removal"},
        "Fight or Flight":  {"cost": 2, "rune": 0, "type": "trick"},
    },
    "Body": {
        "Punch First":      {"cost": 1, "rune": 1, "type": "trick"},
        "Challenge":        {"cost": 1, "rune": 0, "type": "trick"},
        "Confront":         {"cost": 2, "rune": 1, "type": "trick"},
        "Cannon Barrage":   {"cost": 3, "rune": 1, "type": "removal"},
    },
    "Chaos": {
        "Seal of Discord":  {"cost": 1, "rune": 1, "type": "removal"},
        "Rebuke":           {"cost": 2, "rune": 0, "type": "trick"},
        "Stacked Deck":     {"cost": 1, "rune": 0, "type": "draw"},
        "Scrapheap":        {"cost": 1, "rune": 0, "type": "draw"},
        "Last Rites":       {"cost": 2, "rune": 1, "type": "removal"},
    },
    "Calm": {
        "Defy":             {"cost": 1, "rune": 0, "type": "trick"},
        "Discipline":       {"cost": 1, "rune": 0, "type": "trick"},
        "Not So Fast":      {"cost": 2, "rune": 0, "type": "trick"},
        "En Garde":         {"cost": 1, "rune": 0, "type": "trick"},
        "Charm":            {"cost": 2, "rune": 1, "type": "removal"},
        "Wind Wall":        {"cost": 3, "rune": 1, "type": "protection"},
    },
    "Mind": {
        "Stupefy":          {"cost": 2, "rune": 1, "type": "removal"},
        "Wages of Pain":    {"cost": 3, "rune": 1, "type": "removal"},
        "Bellows Breath":   {"cost": 1, "rune": 1, "type": "removal"},
        "Singularity":      {"cost": 3, "rune": 1, "type": "removal"},
        "Time Warp":        {"cost": 5, "rune": 1, "type": "trick"},
    },
}


# ---------------------------------------------------------------------------
# Game History
# ---------------------------------------------------------------------------

class GameHistory:
    """
    Tracks all game events for both players.
    Attached to the GameEngine, queried by strategies.
    """

    def __init__(self):
        # Cards played by each player: {player_name: [card, ...]}
        self.cards_played = defaultdict(list)

        # Specifically track removal/tricks used: {player_name: [card_name, ...]}
        self.removal_used = defaultdict(list)
        self.tricks_used = defaultdict(list)

        # Cards revealed from hand (e.g. from effects): {player_name: [card_name, ...]}
        self.cards_revealed = defaultdict(list)

        # Behavior signals per turn: {player_name: [signal, ...]}
        self.behavior_signals = defaultdict(list)

        # Rune spending log: {player_name: [(turn, amount), ...]}
        self.rune_spending = defaultdict(list)

        # Turn counter
        self.current_turn = 0

    # --- Recording events ---

    def record_card_played(self, player_name: str, card):
        """Called whenever a card is played from hand."""
        self.cards_played[player_name].append(card)

        ability = card.ability.lower()
        name = card.name.lower()

        # Classify what type of card was used
        if card.card_type == "Spell":
            if any(kw in ability for kw in ["deal", "kill", "destroy", "damage"]):
                self.removal_used[player_name].append(card.name)
            elif any(kw in ability for kw in ["ready", "might", "punch", "challenge"]):
                self.tricks_used[player_name].append(card.name)

    def record_passed_with_resources(self, player_name: str, energy: int, runes: int):
        """Called when a player passes/ends turn with resources still available."""
        if energy >= 2 or runes >= 1:
            self.behavior_signals[player_name].append({
                "turn": self.current_turn,
                "type": "passed_with_mana",
                "energy": energy,
                "runes": runes,
            })

    def record_rune_spend(self, player_name: str, amount: int):
        """Track rune spending for tempo analysis."""
        self.rune_spending[player_name].append((self.current_turn, amount))

    # --- Querying knowledge ---

    def removal_count(self, player_name: str) -> int:
        """How many removal spells has this player used?"""
        return len(self.removal_used[player_name])

    def tricks_count(self, player_name: str) -> int:
        """How many combat tricks has this player used?"""
        return len(self.tricks_used[player_name])

    def total_cards_played(self, player_name: str) -> int:
        return len(self.cards_played[player_name])

    def spells_played(self, player_name: str) -> list:
        return [c for c in self.cards_played[player_name] if c.card_type == "Spell"]

    def units_played(self, player_name: str) -> list:
        return [c for c in self.cards_played[player_name] if c.card_type == "Unit"]

    def passed_with_mana_recently(self, player_name: str, within_turns: int = 2) -> bool:
        """Did this player pass with mana up in the last N turns?"""
        for signal in self.behavior_signals[player_name]:
            if signal["type"] == "passed_with_mana":
                if self.current_turn - signal["turn"] <= within_turns:
                    return True
        return False

    def specific_card_used(self, player_name: str, card_name: str) -> int:
        """How many times has the opponent used a specific card?"""
        return sum(1 for c in self.cards_played[player_name] if c.name == card_name)

    def total_runes_spent(self, player_name: str) -> int:
        """Total runes spent across the entire game."""
        return sum(amount for _, amount in self.rune_spending[player_name])

    # --- Threat estimation ---

    def estimate_remaining_threats(self, player_name: str, domains: set) -> dict:
        """
        Estimate what key spells the opponent might still have in deck/hand.

        Based on:
          - Their legend's domains (what pool they draw from)
          - What they've already played (remove used copies)
          - Common deck building (most run 2-3 copies of key spells)

        Returns: {spell_name: {"type": str, "likely_remaining": int, "cost": int, "rune": int}}
        """
        threats = {}
        used_counts = Counter(c.name for c in self.cards_played[player_name])

        for domain in domains:
            if domain not in DOMAIN_KEY_SPELLS:
                continue
            for spell_name, info in DOMAIN_KEY_SPELLS[domain].items():
                used = used_counts.get(spell_name, 0)
                typical_copies = 3   # most decks run 2-3 copies of key spells
                remaining = max(0, typical_copies - used)

                if remaining > 0:
                    threats[spell_name] = {
                        "type": info["type"],
                        "likely_remaining": remaining,
                        "cost": info["cost"],
                        "rune": info["rune"],
                    }

        return threats

    def removal_threat_level(self, player_name: str, domains: set,
                             energy: int, runes: int) -> float:
        """
        How likely is it that the opponent has castable removal right now?

        0.0 = almost impossible (used all copies or can't afford any)
        1.0 = very likely (multiple removal options still available and affordable)
        """
        threats = self.estimate_remaining_threats(player_name, domains)
        castable_removal = []

        for name, info in threats.items():
            if info["type"] != "removal":
                continue
            if info["cost"] <= energy and info["rune"] <= runes:
                castable_removal.append(info)

        if not castable_removal:
            return 0.0

        # More remaining copies = higher threat
        total_remaining = sum(t["likely_remaining"] for t in castable_removal)
        return min(1.0, total_remaining / 5.0)

    def combat_trick_threat(self, player_name: str, domains: set,
                            energy: int, runes: int) -> float:
        """How likely is a combat trick? Same logic as removal_threat_level."""
        threats = self.estimate_remaining_threats(player_name, domains)
        castable_tricks = []

        for name, info in threats.items():
            if info["type"] != "trick":
                continue
            if info["cost"] <= energy and info["rune"] <= runes:
                castable_tricks.append(info)

        if not castable_tricks:
            return 0.0

        total_remaining = sum(t["likely_remaining"] for t in castable_tricks)
        return min(1.0, total_remaining / 5.0)

    # --- Card performance tracking ---

    def record_card_impact(self, player_name: str, card_name: str, impact: str):
        """
        Track when a card makes a meaningful impact during a game.
        impact types: "killed_unit", "killed_champion", "protected", "drew_cards",
                      "buffed", "removed_threat", "scored_point", "wasted"
        """
        if "card_impacts" not in self.__dict__:
            self.card_impacts = {}
        key = (player_name, card_name)
        if key not in self.card_impacts:
            self.card_impacts[key] = {"positive": 0, "negative": 0, "plays": 0}

        self.card_impacts[key]["plays"] += 1
        if impact in ("killed_unit", "killed_champion", "protected", "drew_cards",
                      "buffed", "removed_threat", "scored_point"):
            self.card_impacts[key]["positive"] += 1
        elif impact == "wasted":
            self.card_impacts[key]["negative"] += 1

    def get_card_performance(self, player_name: str) -> dict:
        """
        Get performance stats for all cards played by a player.
        Returns: {card_name: {"plays": N, "positive": N, "negative": N, "score": float}}
        """
        if "card_impacts" not in self.__dict__:
            return {}

        results = {}
        for (pname, card_name), data in self.card_impacts.items():
            if pname != player_name:
                continue
            plays = data["plays"]
            score = (data["positive"] - data["negative"]) / max(plays, 1)
            results[card_name] = {**data, "score": score}

        return results

    # --- Opponent hand modeling ---

    def infer_opponent_hand(self, player_name: str, domains: set,
                            energy: int, runes: int, hand_size: int) -> dict:
        """
        Infer what the opponent is likely holding based on:
          - What they HAVEN'T played (they drew cards but didn't use them)
          - What they passed with resources to cast
          - What removal/tricks they haven't used yet from their domain pool
          - Their hand size (more cards = more options)

        Returns: {
            "likely_has_removal": float (0-1),
            "likely_has_trick": float (0-1),
            "likely_has_champion": float (0-1),
            "likely_holding_for_big_turn": bool,
            "estimated_threat_cards": int,
        }
        """
        inference = {
            "likely_has_removal": 0.0,
            "likely_has_trick": 0.0,
            "likely_has_champion": 0.0,
            "likely_holding_for_big_turn": False,
            "estimated_threat_cards": 0,
        }

        if hand_size == 0:
            return inference

        # Removal likelihood: based on remaining copies in domain pool
        removal_threat = self.removal_threat_level(player_name, domains, energy, runes)
        # Scale by hand size — more cards = higher chance they have it
        hand_factor = min(1.0, hand_size / 4.0)
        inference["likely_has_removal"] = removal_threat * hand_factor

        # Trick likelihood
        trick_threat = self.combat_trick_threat(player_name, domains, energy, runes)
        inference["likely_has_trick"] = trick_threat * hand_factor

        # Champion likelihood: if they've played few champions relative to hand
        champs_played = sum(1 for c in self.cards_played[player_name] if c.champion)
        if champs_played < 2 and hand_size >= 3:
            inference["likely_has_champion"] = 0.5 * hand_factor

        # Holding for big turn: passed with lots of resources multiple turns
        recent_passes = [
            s for s in self.behavior_signals[player_name]
            if s["type"] == "passed_with_mana"
            and self.current_turn - s["turn"] <= 3
        ]
        if len(recent_passes) >= 2:
            # Passed with mana 2+ times recently = sandbagging
            inference["likely_holding_for_big_turn"] = True
            inference["estimated_threat_cards"] = min(hand_size, 3)
        elif recent_passes:
            avg_held_energy = sum(s["energy"] for s in recent_passes) / len(recent_passes)
            if avg_held_energy >= 3:
                inference["likely_holding_for_big_turn"] = True
                inference["estimated_threat_cards"] = min(hand_size, 2)

        return inference

    # --- State encoding for RL ---

    def encode_for_rl(self, player_name: str, opponent_name: str,
                      opp_domains: set, opp_energy: int, opp_runes: int) -> list:
        """
        Encode game history as a flat feature vector for the RL state encoder.
        Returns a list of floats.
        """
        features = []

        # Cards played counts
        features.append(self.total_cards_played(player_name) / 20.0)
        features.append(self.total_cards_played(opponent_name) / 20.0)

        # Removal tracking
        features.append(self.removal_count(opponent_name) / 6.0)
        features.append(self.tricks_count(opponent_name) / 6.0)

        # Threat levels
        features.append(self.removal_threat_level(opponent_name, opp_domains, opp_energy, opp_runes))
        features.append(self.combat_trick_threat(opponent_name, opp_domains, opp_energy, opp_runes))

        # Behavior signals
        features.append(1.0 if self.passed_with_mana_recently(opponent_name) else 0.0)

        # Rune economy
        total_spent = self.total_runes_spent(opponent_name)
        features.append(min(1.0, total_spent / 12.0))

        return features  # 8 features
