"""
ML Agent for Riftbound Sim — Policy Gradient (REINFORCE) with Self-Play

Architecture:
  - State encoder: converts game state into a fixed-size feature vector
  - Policy network: feedforward NN that scores each card in hand
  - Training: self-play games generate (state, action, reward) trajectories;
    REINFORCE updates the network to prefer actions that led to wins

Usage:
  agent = MLAgent(card_pool)
  agent.train(generations=50, games_per_gen=20)
  agent.save("models/policy.pt")

  # Use as a player in GameEngine:
  player = MLPlayer("Agent", deck, agent)
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple

from game.loader import load_card_pool
from game.deck import Deck
from game.player import Player
from game.engine import GameEngine
from ai.genetic import random_genome, genome_to_deck


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

STATE_DIM = 32   # feature vector size

def encode_state(player, opponent, battlefields) -> torch.Tensor:
    """
    Encode current game state as a normalized float vector.

    Features:
      [0]  player score (normalized / 10)
      [1]  opponent score
      [2]  player energy / 10
      [3]  player rune pool / 10
      [4]  hand size / 10
      [5]  deck size / 40
      [6]  base unit count / 5
      [7]  total friendly units on bfs / 10
      [8]  total enemy units on bfs / 10
      [9]  player ahead (1 if score > opp, else 0)
      [10..15] per-battlefield: player units count (3 bfs)
      [13..15] per-battlefield: enemy units count (3 bfs)
      [16..18] per-battlefield: controlled by player (1/0/-1)
      [19]  friendly total Might on board / 20
      [20]  enemy total Might on board / 20
      [21]  turn pressure (turn / 30)
      [22..31] padding zeros
    """
    feats = []

    feats.append(player.score / 10.0)
    feats.append(opponent.score / 10.0)
    feats.append(player.energy / 10.0)
    feats.append(player.rune_pool.pool / 10.0)
    feats.append(len(player.hand) / 10.0)
    feats.append(len(player.deck.cards) / 40.0)
    feats.append(len(player.base_units) / 5.0)

    friendly_all = []
    enemy_all = []
    for bf in battlefields:
        friendly_all.extend(bf.get_units(player.name))
        enemy_all.extend(bf.get_units(opponent.name))

    feats.append(len(friendly_all) / 10.0)
    feats.append(len(enemy_all) / 10.0)
    feats.append(1.0 if player.score > opponent.score else 0.0)

    # Per-battlefield unit counts (3 bfs padded if fewer)
    for i in range(3):
        if i < len(battlefields):
            feats.append(len(battlefields[i].get_units(player.name)) / 5.0)
        else:
            feats.append(0.0)
    for i in range(3):
        if i < len(battlefields):
            feats.append(len(battlefields[i].get_units(opponent.name)) / 5.0)
        else:
            feats.append(0.0)

    # Per-battlefield control signal
    for i in range(3):
        if i < len(battlefields):
            ctrl = battlefields[i].controller
            if ctrl is None:
                feats.append(0.0)
            elif ctrl.name == player.name:
                feats.append(1.0)
            else:
                feats.append(-1.0)
        else:
            feats.append(0.0)

    friendly_might = sum(u.effective_might for u in friendly_all if u.is_alive)
    enemy_might    = sum(u.effective_might for u in enemy_all    if u.is_alive)
    feats.append(friendly_might / 20.0)
    feats.append(enemy_might / 20.0)
    feats.append(0.0)  # turn — filled in externally if available

    # Pad to STATE_DIM
    while len(feats) < STATE_DIM:
        feats.append(0.0)

    return torch.tensor(feats[:STATE_DIM], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Card encoding (for scoring cards in hand)
# ---------------------------------------------------------------------------

CARD_DIM = 8   # features per card

def encode_card(card) -> torch.Tensor:
    """Encode a single card as a feature vector."""
    is_unit  = 1.0 if card.card_type == "Unit"  else 0.0
    is_spell = 1.0 if card.card_type == "Spell" else 0.0
    is_gear  = 1.0 if card.card_type == "Gear"  else 0.0
    return torch.tensor([
        card.cost / 10.0,
        card.rune_cost / 5.0,
        card.health / 10.0,
        card.keyword_value("Assault") / 3.0,
        card.keyword_value("Shield") / 3.0,
        is_unit,
        is_spell,
        is_gear,
    ], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """
    Scores each card in hand given the current game state.

    Input:  state vector (STATE_DIM) + card vector (CARD_DIM)
    Output: scalar score for that card
    """
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + CARD_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, card: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, card], dim=-1)
        return self.net(x).squeeze(-1)

    def score_hand(self, state: torch.Tensor, hand: list) -> torch.Tensor:
        """Return a score tensor for each card in hand."""
        if not hand:
            return torch.tensor([])
        scores = torch.stack([self.forward(state, encode_card(c)) for c in hand])
        return scores


# ---------------------------------------------------------------------------
# ML-driven Player
# ---------------------------------------------------------------------------

Transition = namedtuple("Transition", ["state", "card_idx", "log_prob", "reward"])


class MLPlayer(Player):
    """
    A Player that uses a PolicyNetwork to decide which cards to play.
    Collects (state, action, log_prob) tuples for REINFORCE training.
    """
    def __init__(self, name, deck, policy: PolicyNetwork, training=True):
        super().__init__(name, deck)
        self.policy = policy
        self.training = training
        self.trajectory: list[Transition] = []

    def play_cards(self, battlefields, opponent=None):
        """Override: use policy network to select and order cards to play."""
        affordable = [c for c in self.hand if self.can_afford(c)]
        played_indices = set()

        while True:
            affordable = [c for c in self.hand if self.can_afford(c) and id(c) not in played_indices]
            if not affordable:
                break

            state = encode_state(self, opponent, battlefields) if opponent else torch.zeros(STATE_DIM)
            scores = self.policy.score_hand(state, affordable)

            # Softmax → sample action
            probs = torch.softmax(scores, dim=0)
            dist  = torch.distributions.Categorical(probs)
            idx   = dist.sample().item()
            log_p = dist.log_prob(torch.tensor(idx))

            chosen = affordable[idx]
            played_indices.add(id(chosen))

            if self.training:
                self.trajectory.append(Transition(state, idx, log_p, 0.0))

            # Play the card
            if chosen.card_type == "Unit":
                self.play_unit(chosen, battlefields)
            elif chosen.card_type == "Spell":
                self._play_spell(chosen, battlefields, opponent)
            elif chosen.card_type == "Gear":
                self._play_gear(chosen, battlefields)

            self.hand.remove(chosen)

    def assign_reward(self, reward: float):
        """Called at end of game with win/loss reward (+1 / -1)."""
        self.trajectory = [
            Transition(t.state, t.card_idx, t.log_prob, reward)
            for t in self.trajectory
        ]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MLAgentTrainer:
    """
    Trains a PolicyNetwork via self-play REINFORCE.

    Each generation:
      1. Play N games: ML agent vs random opponent
      2. Collect trajectories
      3. REINFORCE update: maximize E[log_prob * reward]
      4. Log win rate
    """

    def __init__(self, card_pool, hidden=128, lr=1e-3):
        self.card_pool = card_pool
        self.policy = PolicyNetwork(hidden=hidden)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def train(self, generations=100, games_per_gen=20, deck_size=40, verbose=True):
        for gen in range(1, generations + 1):
            all_transitions = []
            wins = 0

            for _ in range(games_per_gen):
                genome = random_genome(self.card_pool, deck_size=deck_size)
                ml_deck  = genome_to_deck(genome, self.card_pool)
                opp_deck = genome_to_deck(
                    random_genome(self.card_pool, deck_size=deck_size),
                    self.card_pool
                )

                ml_player  = MLPlayer("Agent", ml_deck, self.policy, training=True)
                opp_player = Player("Random", opp_deck)

                result = GameEngine(ml_player, opp_player).play_game()
                reward = 1.0 if result == 1 else -1.0

                ml_player.assign_reward(reward)
                all_transitions.extend(ml_player.trajectory)

                if result == 1:
                    wins += 1

            # REINFORCE update
            if all_transitions:
                loss = self._reinforce_loss(all_transitions)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if verbose:
                win_rate = wins / games_per_gen
                print(f"Gen {gen:>4} | Win rate: {win_rate:.2f} | Loss: {loss.item():.4f}")

        return self.policy

    def _reinforce_loss(self, transitions: list) -> torch.Tensor:
        """
        REINFORCE loss: -sum(log_prob * reward)
        Normalize rewards across the batch to reduce variance.
        """
        rewards   = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
        log_probs = torch.stack([t.log_prob for t in transitions])

        # Normalize rewards
        if rewards.std() > 1e-8:
            rewards = (rewards - rewards.mean()) / rewards.std()

        return -(log_probs * rewards).mean()

    def save(self, path="models/policy.pt"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path="models/policy.pt"):
        self.policy.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")


# ---------------------------------------------------------------------------
# Quick entry point
# ---------------------------------------------------------------------------

def train_agent(generations=50, games_per_gen=20, verbose=True):
    """Load card pool and train an ML agent. Returns trained PolicyNetwork."""
    card_pool = load_card_pool()
    trainer = MLAgentTrainer(card_pool)
    policy = trainer.train(
        generations=generations,
        games_per_gen=games_per_gen,
        verbose=verbose
    )
    trainer.save()
    return policy


if __name__ == "__main__":
    train_agent()
