from game.player import Player
from game.battlefield import Battlefield
from game.cards import UnitInstance


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

        bf_names = ["Left", "Right"][:num_battlefields]
        self.battlefields = [
            Battlefield(i, name=name)
            for i, name in enumerate(bf_names)
        ]

    def log(self, message: str):
        if self.verbose:
            print(message)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        self.player1.deck.shuffle()
        self.player2.deck.shuffle()
        self.player1.draw_opening_hand()
        self.player2.draw_opening_hand()

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

    def _sort_damage_targets(self, units: list) -> list:
        """Tank first, Backline last. [Rule 741 / card text]"""
        tank     = [u for u in units if u.has("Tank")]
        normal   = [u for u in units if not u.has("Tank") and not u.has("Backline")]
        backline = [u for u in units if u.has("Backline")]
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
        Resolve combat at one battlefield.
        Both sides deal damage simultaneously.
        Winner takes control of the battlefield.
        """
        atk_units = [u for u in bf.get_units(attacker.name) if not u.is_exhausted]
        def_units = bf.get_units(defender.name)

        if not atk_units:
            return

        if def_units:
            atk_pool = sum(self._effective_might(u, "attacker") for u in atk_units)
            def_pool = sum(self._effective_might(u, "defender") for u in def_units)

            self._assign_damage(atk_pool, self._sort_damage_targets(def_units))
            self._assign_damage(def_pool, self._sort_damage_targets(atk_units))

            self.log(f"  {attacker.name} ({atk_pool}) vs {defender.name} ({def_pool}) at {bf.name}")

            # Deathknell log
            for unit in atk_units + def_units:
                if not unit.is_alive and unit.has("Deathknell"):
                    self.log(f"  [Deathknell] {unit.card.name}")

            # Hunt XP: award XP to attacker for units that died in combat
            for unit in def_units:
                if not unit.is_alive and unit.has("Hunt"):
                    pass  # Hunt awards XP on conquer/hold, not on kill

        else:
            # No defenders — attacker conquers uncontested
            self.log(f"  {attacker.name} conquers {bf.name} uncontested")

        bf.remove_dead_units()

        # Exhaust all attacking units after combat
        for unit in bf.get_units(attacker.name):
            unit.is_exhausted = True

        # Determine new controller
        atk_remaining = bf.get_units(attacker.name)
        def_remaining = bf.get_units(defender.name)

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
                self.log(f"  {defender.name} holds {bf.name} (+{bf.point_value} point)")

        # Ganking: ready units can move to another battlefield next action
        # (simplified: just leave units in place for now)

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

        # Beginning Phase
        active.remove_temporary_units(self.battlefields)
        self.score_holds()

        winner = self.check_winner()
        if winner:
            return winner

        active.start_turn(self.battlefields)

        # Main Phase
        active.play_cards(self.battlefields, opponent)
        active.advance_units_to_battlefields(self.battlefields)

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
