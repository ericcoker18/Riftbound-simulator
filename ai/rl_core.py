"""
Deep RL core for Riftbound — PPO with self-play.

Architecture:
  - Shared backbone encodes the full game state (64 features)
  - 3 policy heads: card selection, deployment, combat
  - 1 value head: predicts win probability
  - PPO (Proximal Policy Optimization) for stable training
  - Self-play: agent plays both sides, learning from every game

Decision flow per turn:
  1. CARD HEAD:   score each card in hand → softmax → sample which to play
  2. DEPLOY HEAD: score each battlefield → softmax → where to place unit
  3. COMBAT HEAD: per-battlefield attack probability → sample attack/hold
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# State encoding (64 features)
# ---------------------------------------------------------------------------

STATE_DIM = 72   # 64 base + 8 history features

def encode_game_state(player, opponent, battlefields, turn=0) -> torch.Tensor:
    """
    Encode the full game state from player's perspective.
    Returns a float tensor of shape (STATE_DIM,).
    """
    f = []

    # --- Global (8) ---
    f.append(player.score / 8.0)
    f.append(opponent.score / 8.0)
    f.append((player.score - opponent.score) / 8.0)   # score delta
    f.append(player.energy / 12.0)
    f.append(player.rune_pool.pool / 10.0)
    f.append(len(player.hand) / 10.0)
    f.append(len(player.deck.cards) / 40.0)
    f.append(turn / 30.0)

    # --- Per-battlefield (3 slots x 12 = 36, padded if < 3 bfs) ---
    for i in range(3):
        if i < len(battlefields):
            bf = battlefields[i]
            fr_units = bf.get_units(player.name)
            en_units = bf.get_units(opponent.name)

            fr_alive = [u for u in fr_units if u.is_alive]
            en_alive = [u for u in en_units if u.is_alive]

            fr_might = sum(u.effective_might for u in fr_alive)
            en_might = sum(u.effective_might for u in en_alive)
            fr_ready = sum(1 for u in fr_alive if not u.is_exhausted)

            ctrl = 0.0
            if bf.controller and bf.controller.name == player.name:
                ctrl = 1.0
            elif bf.controller:
                ctrl = -1.0

            f.append(len(fr_alive) / 5.0)        # friendly count
            f.append(len(en_alive) / 5.0)         # enemy count
            f.append(fr_might / 15.0)             # friendly total might
            f.append(en_might / 15.0)             # enemy total might
            f.append((fr_might - en_might) / 15.0) # might delta
            f.append(fr_ready / 5.0)              # ready units
            f.append(ctrl)                        # control (-1/0/1)
            f.append(bf.point_value / 3.0)        # bf value

            # Champion presence
            fr_champs = sum(1 for u in fr_alive if u.card.champion)
            en_champs = sum(1 for u in en_alive if u.card.champion)
            f.append(fr_champs / 3.0)
            f.append(en_champs / 3.0)

            # Keyword presence (aggregated)
            fr_assault = sum(u.keyword_value("Assault") for u in fr_alive)
            en_tank = sum(1 for u in en_alive if u.has("Tank"))
            f.append(fr_assault / 5.0)
            f.append(en_tank / 3.0)
        else:
            f.extend([0.0] * 12)

    # --- Hand summary (8) ---
    hand = player.hand
    affordable = [c for c in hand if player.can_afford(c)]
    units_in_hand = sum(1 for c in hand if c.card_type == "Unit")
    spells_in_hand = sum(1 for c in hand if c.card_type == "Spell")
    avg_cost = sum(c.cost for c in hand) / max(len(hand), 1)
    max_might = max((c.health for c in hand if c.card_type == "Unit"), default=0)

    f.append(len(affordable) / 5.0)
    f.append(units_in_hand / 5.0)
    f.append(spells_in_hand / 5.0)
    f.append(avg_cost / 10.0)
    f.append(max_might / 10.0)

    # Base units
    f.append(len(player.base_units) / 5.0)
    f.append(len(opponent.base_units) / 5.0)

    # Overall board presence
    total_fr = sum(len(bf.get_units(player.name)) for bf in battlefields)
    f.append(total_fr / 10.0)

    # --- Game history features (8) ---
    history = getattr(player, '_game_history', None)
    if history:
        opp_domains = set()
        d1 = getattr(opponent.rune_pool, 'domain1', None)
        d2 = getattr(opponent.rune_pool, 'domain2', None)
        if d1: opp_domains.add(d1)
        if d2: opp_domains.add(d2)

        hist_feats = history.encode_for_rl(
            player.name, opponent.name,
            opp_domains, opponent.energy, opponent.rune_pool.pool
        )
        f.extend(hist_feats)
    else:
        f.extend([0.0] * 8)

    # --- Pad to STATE_DIM ---
    while len(f) < STATE_DIM:
        f.append(0.0)

    return torch.tensor(f[:STATE_DIM], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Card encoding (8 features per card)
# ---------------------------------------------------------------------------

CARD_DIM = 8

def encode_card(card) -> torch.Tensor:
    return torch.tensor([
        card.cost / 10.0,
        card.rune_cost / 5.0,
        card.health / 10.0,
        card.keyword_value("Assault") / 3.0,
        card.keyword_value("Shield") / 3.0,
        1.0 if card.card_type == "Unit" else 0.0,
        1.0 if card.card_type == "Spell" else 0.0,
        1.0 if card.champion else 0.0,
    ], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Multi-head policy + value network
# ---------------------------------------------------------------------------

class RiftboundNet(nn.Module):
    """
    Shared backbone → 3 policy heads + 1 value head.

    - card_head:   state + card_features → scalar score per card
    - deploy_head: state → 3 scores (one per battlefield)
    - combat_head: state → 3 probabilities (attack at each bf)
    - value_head:  state → predicted win probability
    """

    def __init__(self, hidden=256):
        super().__init__()

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(STATE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Card selection head: state_features + card_features → score
        self.card_head = nn.Sequential(
            nn.Linear(hidden + CARD_DIM, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        # Deployment head: state_features → bf scores
        self.deploy_head = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 3),
        )

        # Combat head: state_features → attack probabilities per bf
        self.combat_head = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 3),
            nn.Sigmoid(),
        )

        # Value head: state_features → win probability
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 1),
            nn.Tanh(),   # output in [-1, 1]: -1=loss, +1=win
        )

    def forward_backbone(self, state: torch.Tensor) -> torch.Tensor:
        return self.backbone(state)

    def score_cards(self, state: torch.Tensor, cards: list) -> torch.Tensor:
        """Score each card in hand. Returns (N,) tensor."""
        if not cards:
            return torch.tensor([])
        features = self.forward_backbone(state)
        scores = []
        for card in cards:
            card_feat = encode_card(card)
            combined = torch.cat([features, card_feat])
            scores.append(self.card_head(combined).squeeze())
        return torch.stack(scores)

    def score_deploy(self, state: torch.Tensor) -> torch.Tensor:
        """Score 3 battlefields for deployment. Returns (3,) tensor."""
        features = self.forward_backbone(state)
        return self.deploy_head(features)

    def combat_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Attack probability per battlefield. Returns (3,) tensor."""
        features = self.forward_backbone(state)
        return self.combat_head(features)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        """Predicted game outcome [-1, 1]. Returns scalar tensor."""
        features = self.forward_backbone(state)
        return self.value_head(features).squeeze()


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """One decision within a game."""
    state: torch.Tensor
    action: int
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: float = 0.0
    decision_type: str = "card"   # "card", "deploy", "combat"


@dataclass
class Trajectory:
    """All steps from one game for one player."""
    steps: list = field(default_factory=list)

    def assign_outcome(self, win: bool):
        """Set final reward for all steps."""
        r = 1.0 if win else -1.0
        for step in self.steps:
            step.reward = r

    def compute_advantages(self, gamma=0.99, lam=0.95):
        """
        GAE (Generalized Advantage Estimation).
        Since all rewards are at game end, this simplifies to:
        advantage = reward - value_prediction
        """
        advantages = []
        returns = []
        for step in self.steps:
            ret = step.reward
            adv = ret - step.value.item()
            advantages.append(adv)
            returns.append(ret)
        return (
            torch.tensor(advantages, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32),
        )
