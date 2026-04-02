import random as _random

from game.player import Player
from game.battlefield import Battlefield
from game.cards import UnitInstance
from game.history import GameHistory


VICTORY_SCORE = 8


class GameEngine:
    def __init__(
        self,
        player1: Player,
        player2: Player,
        max_turns: int = 30,
        num_battlefields: int = 2,
        victory_score: int = VICTORY_SCORE,
        verbose: bool = False,
    ):
        self.player1 = player1
        self.player2 = player2
        self.max_turns = max_turns
        self.victory_score = victory_score
        self.verbose = verbose
        self.turn = 1
        self.is_first_turn_p2 = True
        self.history = GameHistory()

        bf_names = ["Left", "Right"][:num_battlefields]
        self.battlefields = [
            Battlefield(i, name=name)
            for i, name in enumerate(bf_names)
        ]

    def log(self, message: str):
        if self.verbose:
            print(message)

    # ------------------------------------------------------------------
    # Setup: dice roll, draw 4, mulligan
    # ------------------------------------------------------------------

    def setup(self):
        self.player1.deck.shuffle()
        self.player2.deck.shuffle()

        # Dice roll — high roll goes first (winner almost always chooses first)
        roll1 = _random.randint(1, 20)
        roll2 = _random.randint(1, 20)
        while roll1 == roll2:
            roll1 = _random.randint(1, 20)
            roll2 = _random.randint(1, 20)

        if roll2 > roll1:
            # Player 2 won the roll — swap so the winner is always P1
            self.player1, self.player2 = self.player2, self.player1

        self.log(f"Dice roll: {self.player1.name} goes first")

        # Draw 4 cards each
        self.player1.draw_opening_hand(4)
        self.player2.draw_opening_hand(4)

        # Mulligan up to 2 cards (AI decides which to swap)
        self.player1.mulligan(2)
        self.player2.mulligan(2)

    # ------------------------------------------------------------------
    # Keyword helpers (operate on UnitInstance)
    # ------------------------------------------------------------------

    def _effective_might(self, unit: UnitInstance, role: str) -> int:
        """
        Effective Might for combat:
          - Assault N: +N while attacking   [Rule 733]
          - Shield N: +N while defending    [Rule 740]
          - Stun: 0 Might                   [Rule 410]
        """
        if unit.has("Stun"):
            return 0
        base = unit.effective_might
        if role == "attacker":
            base += unit.keyword_value("Assault")
        elif role == "defender":
            base += unit.keyword_value("Shield")
        return max(0, base)

    def _sort_damage_targets(self, units: list, strategy=None) -> list:
        """
        Order targets for damage assignment.

        Rules: Tank units MUST be targeted first, Backline MUST be last.
        Within each tier, the strategy picks the most valuable target to kill.
        Without a strategy, order within tiers is arbitrary.
        """
        tank     = [u for u in units if u.has("Tank")]
        normal   = [u for u in units if not u.has("Tank") and not u.has("Backline")]
        backline = [u for u in units if u.has("Backline")]

        if strategy:
            # Within each tier, prioritize high-value targets:
            # - Champions first (highest value)
            # - Then by killability: prefer units we can actually kill
            #   (waste less overkill damage)
            # - Then by threat: highest Might units are most dangerous alive
            def _target_priority(u):
                score = 0
                if u.card.champion:
                    score += 100          # always kill champions if possible
                score += u.effective_might * 3   # high might = high threat
                if u.has("Assault"):
                    score += u.keyword_value("Assault") * 2
                if u.has("Ganking"):
                    score += 20           # mobile units are dangerous
                return score

            tank     = sorted(tank, key=_target_priority, reverse=True)
            normal   = sorted(normal, key=_target_priority, reverse=True)
            backline = sorted(backline, key=_target_priority, reverse=True)

        return tank + normal + backline

    def _assign_damage(self, pool: int, targets: list):
        """
        Distribute damage across targets in order.
        Lethal must be assigned in full before moving to next unit. [Rule 443.1.d.3]
        Might = base combat strength, not remaining health.
        """
        for unit in targets:
            if pool <= 0:
                break
            lethal = unit.effective_might     # damage needed to kill
            dealt = min(pool, lethal)
            unit.current_health -= dealt
            pool -= dealt

    # ------------------------------------------------------------------
    # Combat at a single battlefield
    # ------------------------------------------------------------------

    def resolve_showdown(self, bf: Battlefield, attacker: Player, defender: Player):
        """
        Resolve combat at one battlefield following the official combat steps:

        1. Combat cleanup — clear temporary combat buffs from previous rounds
        2. Deathknell and damage triggers — deal damage simultaneously, resolve death triggers
        3. Combat results determined — evaluate who won
        4. Control of battlefield established — winner takes control, points awarded
        5. Combat ends — exhaust attackers, final cleanup
        """
        atk_units = [u for u in bf.get_units(attacker.name) if not u.is_exhausted]
        def_units = bf.get_units(defender.name)

        if not atk_units:
            return

        # ===== STEP 1: Combat cleanup =====
        # Clear any temporary combat buffs from previous showdowns this turn
        # (buffs marked as "combat_temp" would be cleared here)
        # Currently buffs persist — this is the hook for future temp buff tracking

        if not def_units:
            # No defenders — uncontested, skip to step 4
            self.log(f"  {attacker.name} conquers {bf.name} uncontested")

        else:
            # ===== STEP 2: Deathknell and damage triggers =====
            # Calculate combat pools
            atk_pool = sum(self._effective_might(u, "attacker") for u in atk_units)
            def_pool = sum(self._effective_might(u, "defender") for u in def_units)

            # Smart damage targeting
            atk_targets = self._sort_damage_targets(def_units, strategy=attacker.strategy)
            def_targets = self._sort_damage_targets(atk_units, strategy=defender.strategy)

            # Deal damage simultaneously
            self._assign_damage(atk_pool, atk_targets)
            self._assign_damage(def_pool, def_targets)

            self.log(f"  {attacker.name} ({atk_pool}) vs {defender.name} ({def_pool}) at {bf.name}")

            # Resolve Deathknell triggers for units that died
            for unit in atk_units + def_units:
                if not unit.is_alive and unit.has("Deathknell"):
                    self.log(f"  [Deathknell] {unit.card.name}")
                    # Deathknell effects would resolve here
                    # (e.g. deal damage, draw cards, spawn tokens)

            # Resolve other damage triggers
            # (units that took damage but survived may have on-damage effects)

        # ===== STEP 3: Combat results determined =====
        # Remove dead units from the battlefield
        bf.remove_dead_units()

        atk_remaining = bf.get_units(attacker.name)
        def_remaining = bf.get_units(defender.name)

        # ===== STEP 4: Control of the battlefield established =====
        if atk_remaining and not def_remaining:
            if bf.controller != attacker:
                bf.controller = attacker
                attacker.score += bf.point_value
                self.log(f"  {attacker.name} conquers {bf.name} (+{bf.point_value} point)")
                # Hunt XP on conquer
                for unit in atk_remaining:
                    if unit.has("Hunt"):
                        attacker.xp += unit.keyword_value("Hunt")
        elif not atk_remaining and def_remaining:
            if bf.controller != defender:
                bf.controller = defender
                defender.score += bf.point_value
                self.log(f"  {defender.name} takes {bf.name} (+{bf.point_value} point)")
        elif not atk_remaining and not def_remaining:
            # Both sides wiped — battlefield becomes uncontrolled
            if bf.controller is not None:
                self.log(f"  {bf.name} is now uncontrolled (mutual destruction)")
                bf.controller = None

        # ===== STEP 5: Combat ends =====
        # Exhaust all attacking units that survived
        for unit in bf.get_units(attacker.name):
            unit.is_exhausted = True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_holds(self):
        """Award points for battlefields held at start of Beginning Phase."""
        for bf in self.battlefields:
            if bf.controller is not None:
                bf.controller.score += bf.point_value
                self.log(f"  {bf.controller.name} holds {bf.name} (+{bf.point_value})")
                # Hunt XP on hold
                for unit in bf.get_units(bf.controller.name):
                    if unit.has("Hunt"):
                        bf.controller.xp += unit.keyword_value("Hunt")

    def check_winner(self):
        if self.player1.score >= self.victory_score:
            return self.player1
        if self.player2.score >= self.victory_score:
            return self.player2
        return None

    # ------------------------------------------------------------------
    # Turn
    # ------------------------------------------------------------------

    def play_turn(self, active: Player, opponent: Player):
        self.log(f"\n--- Turn {self.turn}: {active.name} (score: {active.score}) ---")
        self.history.current_turn = self.turn

        # Give both players access to game history for decision making
        active._game_history = self.history
        opponent._game_history = self.history

        # Beginning Phase
        active.remove_temporary_units(self.battlefields)
        self.score_holds()

        winner = self.check_winner()
        if winner:
            return winner

        # Determine runes to channel
        runes = 2
        if self.is_first_turn_p2 and active == self.player2:
            runes = 3
            self.is_first_turn_p2 = False

        active.start_turn(self.battlefields, runes_to_channel=runes)

        # Main Phase — player records each card played to history
        active.play_cards(self.battlefields, opponent)
        active.advance_units_to_battlefields(self.battlefields)

        # Behavior signal: did the active player end turn with resources up?
        self.history.record_passed_with_resources(
            active.name, active.energy, active.rune_pool.pool
        )

        # Combat Phase — attack only when the strategy says it's favorable
        for bf in self.battlefields:
            atk_units = [u for u in bf.get_units(active.name) if not u.is_exhausted]
            if atk_units:
                if active.strategy:
                    if active.strategy.should_attack(bf, active, opponent):
                        self.resolve_showdown(bf, active, opponent)
                else:
                    self.resolve_showdown(bf, active, opponent)

        self.log(
            f"  Scores: {self.player1.name}={self.player1.score} "
            f"{self.player2.name}={self.player2.score}"
        )

        return self.check_winner()

    # ------------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------------

    def play_game(self) -> int:
        """Returns 1 if player1 wins, 2 if player2 wins, 0 for draw."""
        self.setup()
        # Let players know their opponent for strategy decisions
        self.player1._opponent_name = self.player2.name
        self.player2._opponent_name = self.player1.name

        while self.turn <= self.max_turns:
            winner = self.play_turn(self.player1, self.player2)
            if winner:
                return 1 if winner == self.player1 else 2

            winner = self.play_turn(self.player2, self.player1)
            if winner:
                return 1 if winner == self.player1 else 2

            self.turn += 1

        # Time out — highest score wins
        if self.player1.score > self.player2.score:
            return 1
        elif self.player2.score > self.player1.score:
            return 2
        return 0
