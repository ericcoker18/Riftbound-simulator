import random
from game.deck import Deck
from game.rune_pool import RunePool
from game.cards import UnitInstance
from game.effects import parse_effects, EffectResolver

_resolver = EffectResolver()


class Player:
    def __init__(self, name: str, deck: Deck, domain: str = "Order",
                 domain2: str = None, rune_split: tuple = None,
                 health: int = 20, strategy=None):
        self.name = name
        self.deck = deck
        self.health = health
        self.score = 0
        self.max_energy = 0
        self.energy = 0
        self.hand: list = []
        self.base_units: list = []
        self.rune_pool = RunePool(domain, domain2 or domain, rune_split)
        self.xp = 0
        self.strategy = strategy      # ExpertStrategy or None (basic AI)
        self._opponent_name = None    # set by engine at game start
        self._game_history = None     # set by engine each turn

    def draw_card(self):
        card = self.deck.draw()
        if card is not None:
            self.hand.append(card)

    def draw_opening_hand(self, hand_size: int = 4):
        for _ in range(hand_size):
            self.draw_card()

    def mulligan(self, count: int = 0):
        """
        Put back up to `count` cards from hand and draw replacements.
        AI heuristic: mulligan cards that cost more than 5 (too expensive
        for early game) up to the allowed count.
        """
        if count <= 0 or not self.hand:
            return
        # Find expensive cards to mulligan
        expensive = sorted(
            [c for c in self.hand if c.cost > 5],
            key=lambda c: c.cost, reverse=True
        )
        to_mulligan = expensive[:count]
        for card in to_mulligan:
            self.hand.remove(card)
            self.deck.cards.append(card)
        self.deck.shuffle()
        for _ in range(len(to_mulligan)):
            self.draw_card()

    def start_turn(self, battlefields: list, runes_to_channel: int = 2):
        """
        Start of turn:
        - Gain 1 energy (max 12)
        - Channel runes (normally 2; P2 gets 3 on their first turn)
        - Draw 1 card
        - Ready all units
        """
        self.max_energy = min(self.max_energy + 1, 12)
        self.energy = self.max_energy
        self.rune_pool.channel(runes_to_channel)
        self.draw_card()
        for unit in self.base_units:
            unit.is_exhausted = False
        for bf in battlefields:
            for unit in bf.get_units(self.name):
                unit.is_exhausted = False

    def all_board_units(self, battlefields: list) -> list:
        units = []
        for bf in battlefields:
            units.extend(bf.get_units(self.name))
        return units

    def can_afford(self, card) -> bool:
        return self.energy >= card.cost and self.rune_pool.can_afford(card.rune_cost)

    def _record_play(self, card):
        """Log a played card to game history if available."""
        if self._game_history:
            self._game_history.record_card_played(self.name, card)
            if card.rune_cost > 0:
                self._game_history.record_rune_spend(self.name, card.rune_cost)
            # Track that this card was played (for performance measurement)
            self._game_history.record_card_impact(self.name, card.name, "played")

    def play_unit(self, card, battlefields: list) -> UnitInstance:
        """Pay costs and deploy a unit to the best battlefield."""
        self._record_play(card)
        self.energy -= card.cost
        self.rune_pool.spend(card.rune_cost)

        enters_exhausted = True
        if card.has("Accelerate") and self.rune_pool.can_afford(1):
            self.rune_pool.spend(1)
            enters_exhausted = False

        unit = UnitInstance(card=card, is_exhausted=enters_exhausted)

        best_bf = self._choose_deployment_battlefield(battlefields)
        if best_bf is not None:
            unit.battlefield_id = best_bf.bf_id
            best_bf.add_unit(self.name, unit)
        else:
            unit.battlefield_id = -1
            self.base_units.append(unit)

        return unit

    def _choose_deployment_battlefield(self, battlefields: list):
        """Delegate to strategy if available, else basic heuristic."""
        if self.strategy and self._opponent_name:
            return self.strategy.choose_battlefield(self, self._opponent_name, battlefields)

        # Basic fallback
        opponent_controlled = [
            bf for bf in battlefields
            if bf.controller is not None and bf.controller.name != self.name
        ]
        if opponent_controlled:
            return random.choice(opponent_controlled)
        return min(battlefields, key=lambda bf: len(bf.get_units(self.name)))

    def play_cards(self, battlefields: list, opponent=None):
        """
        Play cards from hand. Uses ExpertStrategy if set, else basic AI.
        """
        if self.strategy and opponent:
            # Expert: strategy decides what to play and in what order
            to_play = self.strategy.choose_cards_to_play(self, battlefields, opponent)
            played = set()
            for card in to_play:
                if self.can_afford(card) and id(card) not in played:
                    played.add(id(card))
                    if card.card_type == "Unit":
                        self.play_unit(card, battlefields)
                    elif card.card_type == "Spell":
                        self._play_spell(card, battlefields, opponent)
                    elif card.card_type == "Gear":
                        self._play_gear(card, battlefields)
            self.hand = [c for c in self.hand if id(c) not in played]
        else:
            # Basic: sort by cost descending, play everything affordable
            self.hand.sort(key=lambda c: (c.cost, c.health), reverse=True)
            remaining = []
            for card in self.hand:
                if self.can_afford(card):
                    if card.card_type == "Unit":
                        self.play_unit(card, battlefields)
                    elif card.card_type == "Spell":
                        self._play_spell(card, battlefields, opponent)
                    elif card.card_type == "Gear":
                        self._play_gear(card, battlefields)
                else:
                    remaining.append(card)
            self.hand = remaining

    def _play_spell(self, card, battlefields: list, opponent):
        self._record_play(card)
        self.energy -= card.cost
        self.rune_pool.spend(card.rune_cost)
        effects = parse_effects(card.ability)
        _resolver.resolve_spell(effects, self, opponent, battlefields,
                                strategy=self.strategy)

    def _play_gear(self, card, battlefields: list):
        self._record_play(card)
        self.energy -= card.cost
        self.rune_pool.spend(card.rune_cost)
        effects = parse_effects(card.ability)
        _resolver.resolve_gear(effects, self, battlefields,
                               strategy=self.strategy)

    def advance_units_to_battlefields(self, battlefields: list):
        still_at_base = []
        for unit in self.base_units:
            if unit.is_exhausted:
                still_at_base.append(unit)
                continue
            target = self._choose_deployment_battlefield(battlefields)
            if target is not None:
                unit.battlefield_id = target.bf_id
                target.add_unit(self.name, unit)
            else:
                still_at_base.append(unit)
        self.base_units = still_at_base

    def remove_dead_base_units(self):
        self.base_units = [u for u in self.base_units if u.is_alive]

    def remove_temporary_units(self, battlefields: list):
        self.base_units = [u for u in self.base_units if not u.has("Temporary")]
        for bf in battlefields:
            alive = [u for u in bf.get_units(self.name) if not u.has("Temporary")]
            bf.set_units(self.name, alive)

    def is_defeated(self) -> bool:
        return self.health <= 0
