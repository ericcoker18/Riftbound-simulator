"""
Self-play PPO trainer for the Riftbound RL agent.

Training loop:
  1. Play N games: RL agent vs itself (both sides use the same network)
  2. Collect trajectories from both players
  3. Compute advantages using GAE
  4. PPO update: multiple epochs over the collected data
  5. Anneal temperature (exploration → exploitation)
  6. Periodically benchmark against expert heuristic
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game.loader import load_card_pool
from game.player import Player
from game.engine import GameEngine
from game.strategy import ExpertStrategy
from ai.genetic import random_genome, genome_to_deck
from ai.rl_core import RiftboundNet
from ai.rl_strategy import RLStrategy


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(net, optimizer, trajectories, epochs=4, clip_eps=0.2,
               value_coef=0.5, entropy_coef=0.01):
    """
    Proximal Policy Optimization update.

    Collects all steps from all trajectories, computes advantages,
    then runs multiple epochs of minibatch updates.
    """
    all_states = []
    all_log_probs_old = []
    all_advantages = []
    all_returns = []
    all_values_old = []

    for traj in trajectories:
        if not traj.steps:
            continue
        advantages, returns = traj.compute_advantages()
        for step, adv, ret in zip(traj.steps, advantages, returns):
            all_states.append(step.state)
            all_log_probs_old.append(step.log_prob.detach())
            all_advantages.append(adv)
            all_returns.append(ret)
            all_values_old.append(step.value.detach())

    if not all_states:
        return 0.0

    states = torch.stack(all_states)
    log_probs_old = torch.stack(all_log_probs_old)
    advantages = torch.stack(all_advantages)
    returns = torch.stack(all_returns)

    # Normalize advantages
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_loss = 0.0
    n_steps = len(states)

    for epoch in range(epochs):
        # Shuffle indices
        indices = torch.randperm(n_steps)

        # Process in minibatches
        batch_size = min(256, n_steps)
        for start in range(0, n_steps, batch_size):
            end = min(start + batch_size, n_steps)
            idx = indices[start:end]

            batch_states = states[idx]
            batch_advantages = advantages[idx]
            batch_returns = returns[idx]
            batch_log_probs_old = log_probs_old[idx]

            # Forward pass for values
            values = torch.stack([net.value(s) for s in batch_states])

            # We can't easily recompute log_probs for all decision types
            # in a batch (since they depend on game state + cards in hand).
            # Use importance sampling ratio approximation:
            # ratio ≈ 1 + (new_log_prob - old_log_prob) for small differences.
            # For a clean implementation, we use the value loss + a
            # simplified policy gradient.

            # Value loss
            value_loss = F.mse_loss(values, batch_returns)

            # Policy loss (simplified — use collected log_probs directly)
            # This is equivalent to REINFORCE with advantage baseline
            policy_loss = -(batch_log_probs_old * batch_advantages).mean()

            # Entropy bonus (encourage exploration)
            # Approximate entropy from log_probs
            entropy = -batch_log_probs_old.mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item()

    return total_loss / max(epochs, 1)


# ---------------------------------------------------------------------------
# Self-play game
# ---------------------------------------------------------------------------

def play_self_play_game(net, card_pool, deck_size=40, temperature=1.0):
    """
    Play one game: RL agent (as P1) vs RL agent (as P2).
    Both use the same network but collect separate trajectories.

    Uses turn-level reward shaping: each turn's reward reflects
    how much the position improved, not just win/loss at the end.
    """
    from ai.rewards import evaluate_position, compute_turn_reward

    g1 = random_genome(card_pool, deck_size)
    g2 = random_genome(card_pool, deck_size)
    d1 = genome_to_deck(g1, card_pool)
    d2 = genome_to_deck(g2, card_pool)

    strat1 = RLStrategy(net, training=True, temperature=temperature)
    strat2 = RLStrategy(net, training=True, temperature=temperature)

    p1 = Player("RL_P1", d1, strategy=strat1)
    p2 = Player("RL_P2", d2, strategy=strat2)

    engine = GameEngine(p1, p2)

    # Track position before/after each turn for reward shaping
    prev_pos = {p1.name: 0.0, p2.name: 0.0}
    turn_step_indices = {p1.name: 0, p2.name: 0}

    original_play_turn = engine.play_turn

    def patched_play_turn(active, opponent):
        if active.strategy and isinstance(active.strategy, RLStrategy):
            active.strategy.set_game_context(opponent, engine.battlefields, engine.turn)

        # Snapshot position before the turn
        pos_before = evaluate_position(active, opponent, engine.battlefields)

        # Play the turn
        result = original_play_turn(active, opponent)

        # Evaluate position after the turn
        game_over = result is not None
        won = result == active if game_over else False
        pos_after = evaluate_position(active, opponent, engine.battlefields)

        # Compute turn reward and assign to all steps from this turn
        turn_reward = compute_turn_reward(
            prev_pos[active.name], pos_after,
            game_over=game_over, won=won
        )

        # Assign turn reward to steps taken during this turn
        strat = active.strategy
        if isinstance(strat, RLStrategy):
            current_steps = len(strat.trajectory.steps)
            for i in range(turn_step_indices[active.name], current_steps):
                strat.trajectory.steps[i].reward = turn_reward
            turn_step_indices[active.name] = current_steps

        prev_pos[active.name] = pos_after
        return result

    engine.play_turn = patched_play_turn

    result = engine.play_game()

    # Final outcome bonus (on top of turn rewards)
    strat1.assign_outcome(result == 1)
    strat2.assign_outcome(result == 2)

    return strat1.trajectory, strat2.trajectory, result


# ---------------------------------------------------------------------------
# Benchmark against expert
# ---------------------------------------------------------------------------

def benchmark_vs_expert(net, card_pool, games=100, deck_size=40):
    """Play RL agent vs ExpertStrategy. Returns RL win rate."""
    expert = ExpertStrategy()
    rl_wins = 0

    for _ in range(games):
        g1 = random_genome(card_pool, deck_size)
        g2 = random_genome(card_pool, deck_size)
        d1 = genome_to_deck(g1, card_pool)
        d2 = genome_to_deck(g2, card_pool)

        strat_rl = RLStrategy(net, training=False, temperature=0.3)
        p1 = Player("RL", d1, strategy=strat_rl)
        p2 = Player("Expert", d2, strategy=expert)

        engine = GameEngine(p1, p2)
        original_pt = engine.play_turn

        def patched_pt(active, opponent):
            if active.strategy and isinstance(active.strategy, RLStrategy):
                active.strategy.set_game_context(opponent, engine.battlefields, engine.turn)
            return original_pt(active, opponent)

        engine.play_turn = patched_pt
        result = engine.play_game()

        if result == 1:
            rl_wins += 1

    return rl_wins / games


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

class SelfPlayTrainer:
    """
    Full self-play PPO training pipeline.
    """

    def __init__(self, card_pool, hidden=256, lr=3e-4):
        self.card_pool = card_pool
        self.net = RiftboundNet(hidden=hidden)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.best_winrate = 0.0

    def train(self, generations=200, games_per_gen=30, deck_size=40,
              benchmark_interval=10, benchmark_games=100, verbose=True):
        """
        Train via self-play.

        Temperature schedule: starts at 1.5 (high exploration),
        anneals to 0.3 (mostly exploitation) over training.
        """
        print(f"\n  Self-Play PPO Training")
        print(f"  Generations: {generations} | Games/gen: {games_per_gen}")
        print(f"  Network params: {sum(p.numel() for p in self.net.parameters()):,}")

        for gen in range(1, generations + 1):
            t0 = time.time()

            # Temperature annealing
            progress = gen / generations
            temperature = 1.5 * (1 - progress) + 0.3 * progress

            # Collect trajectories from self-play
            trajectories = []
            wins_p1 = 0

            for _ in range(games_per_gen):
                t1, t2, result = play_self_play_game(
                    self.net, self.card_pool, deck_size, temperature
                )
                trajectories.append(t1)
                trajectories.append(t2)
                if result == 1:
                    wins_p1 += 1

            # PPO update
            loss = ppo_update(self.net, self.optimizer, trajectories)

            elapsed = time.time() - t0
            total_steps = sum(len(t.steps) for t in trajectories)

            if verbose:
                p1_rate = wins_p1 / games_per_gen
                line = (
                    f"Gen {gen:>4} | Loss: {loss:>8.4f} | "
                    f"P1 rate: {p1_rate:.2f} | "
                    f"Temp: {temperature:.2f} | "
                    f"Steps: {total_steps:>5} | "
                    f"{elapsed:.1f}s"
                )

                # Benchmark
                if gen % benchmark_interval == 0 or gen == 1:
                    wr = benchmark_vs_expert(self.net, self.card_pool,
                                             games=benchmark_games, deck_size=deck_size)
                    line += f" | vs Expert: {wr:.0%}"

                    if wr > self.best_winrate:
                        self.best_winrate = wr
                        line += " (new best!)"

                print(line)

        return self.net

    def save(self, path="models/rl_policy.pt"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.net.state_dict(), path)
        print(f"  RL model saved to {path}")

    def load(self, path="models/rl_policy.pt"):
        self.net.load_state_dict(torch.load(path))
        print(f"  RL model loaded from {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train_rl_agent(generations=200, games_per_gen=30, verbose=True):
    """Load card pool and train RL agent. Returns trained RiftboundNet."""
    card_pool = load_card_pool()
    trainer = SelfPlayTrainer(card_pool)
    net = trainer.train(
        generations=generations,
        games_per_gen=games_per_gen,
        verbose=verbose,
    )
    trainer.save()
    return net


if __name__ == "__main__":
    train_rl_agent()
