"""
RL-powered strategy that uses the trained RiftboundNet for all decisions.

Falls back to ExpertStrategy when the network isn't confident or
during early training when the network hasn't learned enough.
"""

import torch
import random

from game.strategy import ExpertStrategy, card_play_score
from ai.rl_core import encode_game_state, encode_card, RiftboundNet, Step, Trajectory, DEVICE


class RLStrategy(ExpertStrategy):
    """
    Subclasses ExpertStrategy so all expert heuristics are the fallback.
    Overrides each decision point to use the neural network when training
    or when the network has been trained.
    """

    def __init__(self, net: RiftboundNet, training=True, temperature=1.0):
        super().__init__()
        self.net = net
        self.training = training
        self.temperature = temperature   # exploration: higher = more random
        self.trajectory = Trajectory()

        # Set during gameplay
        self._opponent = None
        self._battlefields = None
        self._turn = 0

    def set_game_context(self, opponent, battlefields, turn):
        """Called by the engine each turn to update context."""
        self._opponent = opponent
        self._battlefields = battlefields
        self._turn = turn

    def _get_state(self, player):
        if self._opponent and self._battlefields:
            return encode_game_state(player, self._opponent, self._battlefields, self._turn)
        return torch.zeros(64, device=DEVICE)

    # --- Card selection ---

    def choose_cards_to_play(self, player, battlefields, opponent):
        """
        Use the card head to score each affordable card.
        Sample which to play using softmax probabilities.
        Falls back to expert for resource holdback logic.
        """
        self._opponent = opponent
        self._battlefields = battlefields

        affordable = [c for c in player.hand if player.can_afford(c)]
        if not affordable:
            return []

        state = self._get_state(player)

        # Score all affordable cards with the network
        with torch.no_grad() if not self.training else torch.enable_grad():
            scores = self.net.score_cards(state, affordable)
            value = self.net.value(state)

        # Add "pass" option (score 0)
        pass_score = torch.tensor([0.0], device=DEVICE)
        all_scores = torch.cat([scores, pass_score])

        # Temperature-scaled softmax for exploration
        probs = torch.softmax(all_scores / max(self.temperature, 0.1), dim=0)
        dist = torch.distributions.Categorical(probs)

        to_play = []
        sim_energy = player.energy
        sim_runes = player.rune_pool.pool
        played_ids = set()

        # Sample cards to play one at a time
        for _ in range(len(affordable)):
            remaining = [
                (i, c) for i, c in enumerate(affordable)
                if id(c) not in played_ids
                and sim_energy >= c.cost
                and sim_runes >= c.rune_cost
            ]
            if not remaining:
                break

            # Re-score remaining cards
            rem_cards = [c for _, c in remaining]
            rem_scores = self.net.score_cards(state, rem_cards) if self.training else scores[[i for i, _ in remaining]]
            pass_s = torch.tensor([0.0], device=DEVICE)
            all_s = torch.cat([rem_scores, pass_s])
            probs = torch.softmax(all_s / max(self.temperature, 0.1), dim=0)
            dist = torch.distributions.Categorical(probs)

            idx = dist.sample().item()

            if idx == len(rem_cards):
                # Chose to pass — stop playing
                if self.training:
                    self.trajectory.steps.append(Step(
                        state=state, action=idx, log_prob=dist.log_prob(torch.tensor(idx, device=DEVICE)),
                        value=value, decision_type="card_pass"
                    ))
                break

            chosen = rem_cards[idx]
            played_ids.add(id(chosen))
            to_play.append(chosen)
            sim_energy -= chosen.cost
            sim_runes -= chosen.rune_cost

            if self.training:
                self.trajectory.steps.append(Step(
                    state=state, action=idx, log_prob=dist.log_prob(torch.tensor(idx, device=DEVICE)),
                    value=value, decision_type="card"
                ))

        return to_play

    # --- Deployment ---

    def choose_battlefield(self, player, opponent_name, battlefields):
        """Use deploy head to pick the best battlefield."""
        if not battlefields:
            return None

        state = self._get_state(player)

        with torch.no_grad() if not self.training else torch.enable_grad():
            scores = self.net.score_deploy(state)
            value = self.net.value(state)

        # Mask out battlefields beyond what exists
        n_bfs = min(len(battlefields), 3)
        valid_scores = scores[:n_bfs]

        probs = torch.softmax(valid_scores / max(self.temperature, 0.1), dim=0)
        dist = torch.distributions.Categorical(probs)
        idx = dist.sample().item()

        if self.training:
            self.trajectory.steps.append(Step(
                state=state, action=idx, log_prob=dist.log_prob(torch.tensor(idx, device=DEVICE)),
                value=value, decision_type="deploy"
            ))

        return battlefields[idx]

    # --- Combat ---

    def should_attack(self, bf, attacker, defender):
        """Use combat head for attack/hold decision."""
        state = self._get_state(attacker)

        with torch.no_grad() if not self.training else torch.enable_grad():
            probs = self.net.combat_probs(state)
            value = self.net.value(state)

        bf_idx = min(bf.bf_id, 2)
        attack_prob = probs[bf_idx].item()

        # Sample decision
        attack = random.random() < attack_prob

        if self.training:
            # Log prob for the chosen action
            p = attack_prob if attack else (1 - attack_prob)
            log_p = torch.log(torch.tensor(max(p, 1e-8), device=DEVICE))
            self.trajectory.steps.append(Step(
                state=state, action=1 if attack else 0,
                log_prob=log_p, value=value, decision_type="combat"
            ))

        return attack

    # --- Spell targeting (delegate to expert with network tie-breaking) ---
    # For now, spell targeting uses the expert heuristic.
    # The value head still improves overall play quality since it influences
    # card selection (when to play removal) and combat (when to attack).

    def assign_outcome(self, win: bool):
        """Called at end of game to set rewards."""
        self.trajectory.assign_outcome(win)

    def reset(self):
        """Reset trajectory for a new game."""
        self.trajectory = Trajectory()
        self._opponent = None
        self._battlefields = None
        self._turn = 0
