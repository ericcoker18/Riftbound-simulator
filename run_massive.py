"""
Massive simulation run to find the absolute best meta deck.

Pipeline:
  1. Load real meta decks from data/meta_decks.json (tournament baselines)
  2. Train (or load) ML policy via self-play
  3. Evolve a large population across many generations using all CPU cores
  4. Detect convergence: stop when the top deck stabilizes
  5. Print the final absolute best deck

Usage:
  python run_massive.py

Flags (edit CONFIG below or pass as env vars):
  RETRAIN=1        Force retrain the ML policy even if a saved one exists
  WORKERS=N        Override number of parallel workers (default: all cores)
"""

import os
import time
import random
import multiprocessing
from collections import Counter

import json

from game.loader import load_card_pool
from game.legend import load_legends
from ai.genetic import (
    random_genome, genome_to_deck, genome_legend, genome_cards,
    summarize_deck, crossover, mutate, select, update_hall_of_fame,
    fitness_vs_pool, get_legend,
)
from ai.parallel import ParallelEvaluator
from ai.ml_agent import MLAgentTrainer
from ai.self_play import SelfPlayTrainer, benchmark_vs_expert

STATUS_PATH = "results/sim_status.json"

def write_status(phase, detail="", progress=0.0, gen=0, max_gen=0, extra=None):
    """Write simulation status to a JSON file for the dashboard to read."""
    os.makedirs("results", exist_ok=True)
    status = {
        "phase": phase,
        "detail": detail,
        "progress": progress,
        "generation": gen,
        "max_generation": max_gen,
        "timestamp": time.time(),
    }
    if extra:
        status.update(extra)
    with open(STATUS_PATH, "w") as f:
        json.dump(status, f)

def clear_status():
    if os.path.exists(STATUS_PATH):
        os.remove(STATUS_PATH)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # ML training (basic REINFORCE)
    "ml_model_path":     "models/policy.pt",
    "ml_generations":    100,
    "ml_games_per_gen":  30,

    # RL self-play (PPO)
    "rl_model_path":     "models/rl_policy.pt",
    "rl_generations":    200,
    "rl_games_per_gen":  30,

    # GA evolution
    "deck_size":         40,
    "population_size":   100,
    "max_generations":   200,
    "top_n":             30,
    "mutation_rate":     0.08,
    "opponent_pool_size": 20,
    "games_per_opponent": 15,
    "hall_of_fame_size":  20,

    # Opponent mix (must sum to <= 1.0; remainder is random)
    "coevo_ratio":       0.25,   # Hall of Fame opponents
    "ml_ratio":          0.25,   # ML agent opponents
    "meta_ratio":        0.20,   # Tournament meta deck opponents

    # Convergence: stop if top deck composition unchanged for N generations
    "convergence_window": 15,
    "convergence_threshold": 0.95,  # 95% card overlap = converged
}


# ---------------------------------------------------------------------------
# Meta deck loader
# ---------------------------------------------------------------------------

# Map tournament legend names to data/legends.json names
LEGEND_ALIAS = {
    "Garen":       "Garen - Might of Demacia",
    "Lee Sin":     "Lee Sin - Blind Monk",
    "Rumble":      "Rumble - Mechanized Menace",
    "Leona":       "Leona - Radiant Dawn",
    "Renata Glasc":"Renata Glasc - Chem-Baroness",
    "Yasuo":       "Yasuo - Unforgiven",
    "Miss Fortune":"Miss Fortune - Bounty Hunter",
    "Sett":        "Sett - The Boss",
    "Darius":      "Darius - Hand of Noxus",
    "Volibear":    "Volibear - Relentless Storm",
    "Ahri":        "Ahri - Nine-Tailed Fox",
    "Fiora":       "Fiora - Grand Duelist",
    "Annie":       "Annie - Dark Child",
    "Lux":         "Lux - Lady of Luminosity",
    "Teemo":       "Teemo - Swift Scout",
    "Lucian":      "Lucian - Purifier",
    "Ornn":        "Ornn - Fire Below the Mountain",
    "Rek'Sai":     "Rek'sai - Void Burrower",
    "Jinx":        "Jinx - Loose Cannon",
    "Master Yi":   "Master Yi - Wuju Bladesman",
    "Viktor":      "Viktor - Herald of the Arcane",
    "Azir":        "Azir - Emperor of the Sands",
    "Kai'Sa":      "Kai'Sa - Daughter of the Void",
    "Sivir":       "Sivir - Battle Mistress",
    "Draven":      "Draven - Glorious Executioner",
    "Ezreal":      "Ezreal - Prodigal Explorer",
    "Irelia":      "Irelia - Blade Dancer",
    "Jax":         "Jax - Grandmaster At Arms",
}


def load_meta_decks(path="data/meta_decks.json", card_pool=None):
    """
    Load tournament-winning decklists from data/meta_decks.json.
    Returns a list of genomes in (legend_name, card_list) format.
    """
    import json

    if not os.path.exists(path):
        return []

    pool_names = {c.name for c in card_pool} if card_pool else None

    # Build alias map for card name normalization
    alias_map = {}
    if pool_names:
        for pn in pool_names:
            alias_map[pn] = pn
            alias_map[pn.lower()] = pn
            if " - " in pn:
                comma_form = pn.replace(" - ", ", ")
                alias_map[comma_form] = pn
                alias_map[comma_form.lower()] = pn

    # Load legend names for lookup
    legends_by_name = {}
    try:
        for l in load_legends():
            legends_by_name[l.name] = l
    except Exception:
        pass

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    genomes = []
    for entry in raw:
        name    = entry.get("name", "Unknown")
        cards   = entry.get("cards", [])
        genome  = []
        skipped = []

        for slot in cards:
            card_name = slot["name"]
            resolved  = alias_map.get(card_name) or alias_map.get(card_name.lower()) if alias_map else card_name
            count     = slot.get("count", 1)
            if resolved is None:
                skipped.append(card_name)
                continue
            genome.extend([resolved] * count)

        if skipped:
            print(f"  [meta] '{name}': skipped {len(skipped)} unrecognized cards: {skipped[:3]}...")

        # Detect legend from deck name (e.g. "Eric - Jinx" -> "Jinx")
        legend_name = None
        for alias, full_name in LEGEND_ALIAS.items():
            if alias.lower() in name.lower():
                legend_name = full_name
                break

        if genome:
            if legend_name:
                print(f"  [meta] Loaded '{name}' ({len(genome)} cards, {legend_name})")
                genomes.append((legend_name, genome))
            else:
                # Fallback: wrap with unknown legend — still usable as opponent
                print(f"  [meta] Loaded '{name}' ({len(genome)} cards, legend unknown)")
                genomes.append(("Unknown", genome))

    return genomes


# ---------------------------------------------------------------------------
# Convergence tracker
# ---------------------------------------------------------------------------

class ConvergenceTracker:
    """
    Tracks the top deck across generations.
    Declares convergence when card overlap stays >= threshold for `window` gens.
    """

    def __init__(self, window=15, threshold=0.95):
        self.window = window
        self.threshold = threshold
        self.history = []

    def update(self, genome) -> bool:
        """Add this generation's best genome. Returns True if converged."""
        cards = genome_cards(genome)
        self.history.append(Counter(cards))
        if len(self.history) < self.window:
            return False

        recent = self.history[-self.window:]
        ref = recent[-1]
        ref_total = sum(ref.values())

        similarities = []
        for past in recent[:-1]:
            overlap = sum(min(ref[k], past[k]) for k in ref)
            similarities.append(overlap / max(ref_total, 1))

        return (sum(similarities) / len(similarities)) >= self.threshold

    def stability(self) -> float:
        if len(self.history) < 2:
            return 0.0
        recent = self.history[-min(self.window, len(self.history)):]
        ref = recent[-1]
        ref_total = sum(ref.values())
        sims = []
        for past in recent[:-1]:
            overlap = sum(min(ref[k], past[k]) for k in ref)
            sims.append(overlap / max(ref_total, 1))
        return sum(sims) / len(sims) if sims else 0.0


# ---------------------------------------------------------------------------
# Meta analytics tracker
# ---------------------------------------------------------------------------

class MetaTracker:
    """
    Collects statistics across the entire evolution run:
      - Which legends win the most (appear as gen winners)
      - Which champion pairings win with each legend
      - Which cards appear most frequently in top decks
    """

    def __init__(self, card_pool):
        self.card_pool = card_pool
        self.pool_lookup = {c.name: c for c in card_pool}

        # Per-generation winners
        self.legend_wins = Counter()           # legend_name -> count of gen wins
        self.legend_scores = {}                # legend_name -> list of scores
        self.own_champion_wins = Counter()     # (legend, own_champ) -> count (Champion zone)
        self.support_champion_wins = Counter() # (legend, other_champ) -> count (in-deck)
        self.card_frequency = Counter()        # card_name -> total appearances in top decks
        self.top_deck_count = 0

        # Per-generation: track the top N decks
        self.all_top_decks = []                # list of (score, genome) across all gens

    def record_generation(self, population, scores, top_n=5):
        """Record stats from one generation's results."""
        ranked = sorted(zip(scores, population), reverse=True)

        # Top deck = generation winner
        top_score, top_genome = ranked[0]
        legend_name = genome_legend(top_genome)
        self.legend_wins[legend_name] += 1

        if legend_name not in self.legend_scores:
            self.legend_scores[legend_name] = []
        self.legend_scores[legend_name].append(top_score)

        # Separate own champions (Champion zone) from other champions (support)
        try:
            legend_obj = get_legend(legend_name)
            own_champ_names = {c.name for c in legend_obj.get_own_champions(self.card_pool)}
        except (KeyError, Exception):
            own_champ_names = set()

        cards = genome_cards(top_genome)
        for name in cards:
            card = self.pool_lookup.get(name)
            if card and card.champion:
                if name in own_champ_names:
                    self.own_champion_wins[(legend_name, name)] += 1
                else:
                    self.support_champion_wins[(legend_name, name)] += 1

        # Track card frequency across top N decks this generation
        for _, genome in ranked[:top_n]:
            self.top_deck_count += 1
            for name in genome_cards(genome):
                self.card_frequency[name] += 1

        self.all_top_decks.append((top_score, top_genome))

    def report(self):
        """Print the full meta analytics report."""
        print(f"\n{'='*60}")
        print("  META ANALYTICS REPORT")
        print(f"{'='*60}")

        # --- Legend win rates ---
        print(f"\n  LEGEND DOMINANCE (generation wins)")
        print(f"  {'Legend':<35} {'Wins':>6} {'Avg Score':>10}")
        print(f"  {'-'*35} {'-'*6} {'-'*10}")
        for legend, wins in self.legend_wins.most_common():
            avg = sum(self.legend_scores[legend]) / len(self.legend_scores[legend])
            bar = "#" * min(wins, 40)
            print(f"  {legend:<35} {wins:>6} {avg:>10.3f}  {bar}")

        # --- Champion zone (legend's own champion card) ---
        print(f"\n  CHAMPION ZONE (legend's own champion card)")
        print(f"  {'Legend':<30} {'Champion':<30} {'Wins':>6}")
        print(f"  {'-'*30} {'-'*30} {'-'*6}")
        for (legend, champ), wins in self.own_champion_wins.most_common(20):
            print(f"  {legend:<30} {champ:<30} {wins:>6}")

        # --- Best supporting champion units ---
        print(f"\n  BEST SUPPORTING CHAMPIONS (other champion units in deck)")
        print(f"  {'Legend':<30} {'Support Champion':<30} {'Wins':>6}")
        print(f"  {'-'*30} {'-'*30} {'-'*6}")
        for (legend, champ), wins in self.support_champion_wins.most_common(20):
            print(f"  {legend:<30} {champ:<30} {wins:>6}")

        # --- Most frequent cards ---
        print(f"\n  MOST FREQUENT CARDS IN TOP DECKS ({self.top_deck_count} decks sampled)")
        print(f"  {'Card':<35} {'Type':<8} {'Domain':<8} {'Appearances':>12} {'Rate':>8}")
        print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*12} {'-'*8}")
        for name, count in self.card_frequency.most_common(40):
            card = self.pool_lookup.get(name)
            if card:
                rate = count / self.top_deck_count
                ctype = card.card_type[:7]
                domain = (card.domain or "?")[:7]
                bar = "#" * min(int(rate * 20), 20)
                print(f"  {name:<35} {ctype:<8} {domain:<8} {count:>12} {rate:>7.0%}  {bar}")

        # --- Card type breakdown across top decks ---
        type_counts = Counter()
        for name, count in self.card_frequency.items():
            card = self.pool_lookup.get(name)
            if card:
                type_counts[card.card_type] += count
        total = sum(type_counts.values())
        if total:
            print(f"\n  CARD TYPE MIX IN TOP DECKS")
            for ctype, count in type_counts.most_common():
                pct = count / total * 100
                print(f"  {ctype:<12} {pct:>5.1f}%")

        # --- Domain breakdown across top decks ---
        domain_counts = Counter()
        for name, count in self.card_frequency.items():
            card = self.pool_lookup.get(name)
            if card and card.domain:
                domain_counts[card.domain] += count
        total = sum(domain_counts.values())
        if total:
            print(f"\n  DOMAIN MIX IN TOP DECKS")
            for domain, count in domain_counts.most_common():
                pct = count / total * 100
                print(f"  {domain:<12} {pct:>5.1f}%")


# ---------------------------------------------------------------------------
# Core evolution loop (parallel)
# ---------------------------------------------------------------------------

def evolve_massive(card_pool, ml_policy=None, meta_genomes=None, cfg=CONFIG):
    deck_size         = cfg["deck_size"]
    population_size   = cfg["population_size"]
    max_generations   = cfg["max_generations"]
    top_n             = cfg["top_n"]
    mutation_rate     = cfg["mutation_rate"]
    opponent_pool_size = cfg["opponent_pool_size"]
    games_per_opponent = cfg["games_per_opponent"]
    hof_size          = cfg["hall_of_fame_size"]
    coevo_ratio       = cfg["coevo_ratio"]
    ml_ratio          = cfg["ml_ratio"] if ml_policy else 0.0
    meta_ratio        = cfg["meta_ratio"] if meta_genomes else 0.0

    meta_genomes = meta_genomes or []

    population   = [random_genome(card_pool, deck_size) for _ in range(population_size)]
    hall_of_fame = []
    tracker      = ConvergenceTracker(cfg["convergence_window"], cfg["convergence_threshold"])
    meta_tracker = MetaTracker(card_pool)

    best_genome = None
    best_score  = -1

    print(f"\n  Population : {population_size}")
    print(f"  Opponent mix: {int((1-coevo_ratio-ml_ratio-meta_ratio)*100)}% random | "
          f"{int(coevo_ratio*100)}% HoF | "
          f"{int(ml_ratio*100)}% ML | "
          f"{int(meta_ratio*100)}% meta")
    print(f"  Max generations: {max_generations} (early stop on convergence)")

    num_workers = int(os.environ.get("WORKERS", multiprocessing.cpu_count()))

    with ParallelEvaluator(card_pool, ml_policy, num_workers=num_workers) as evaluator:
        for gen in range(max_generations):
            t0 = time.time()

            # Build opponent pool
            n_meta   = int(opponent_pool_size * meta_ratio)
            n_ml     = int(opponent_pool_size * ml_ratio)
            n_hof    = int(opponent_pool_size * coevo_ratio) if hall_of_fame else 0
            n_random = opponent_pool_size - n_meta - n_ml - n_hof

            random_opps = [(random_genome(card_pool, deck_size), False) for _ in range(max(n_random, 0))]
            hof_opps    = [(random.choice(hall_of_fame), False) for _ in range(n_hof)] if hall_of_fame and n_hof > 0 else []
            ml_opps     = [(random_genome(card_pool, deck_size), True)  for _ in range(n_ml)]
            meta_opps   = [(random.choice(meta_genomes), False)         for _ in range(n_meta)] if meta_genomes and n_meta > 0 else []

            opponent_pool     = random_opps + hof_opps + ml_opps + meta_opps
            expanded_opponents = opponent_pool * games_per_opponent

            # Parallel fitness evaluation
            scores = evaluator.evaluate(population, expanded_opponents)

            top_score  = max(scores)
            top_genome = population[scores.index(top_score)]

            if top_score > best_score:
                best_score  = top_score
                best_genome = top_genome

            update_hall_of_fame(hall_of_fame, top_genome, hof_size)
            converged = tracker.update(top_genome)
            meta_tracker.record_generation(population, scores, top_n=min(5, population_size))

            elapsed  = time.time() - t0
            n_games  = len(population) * len(expanded_opponents)
            gps      = n_games / elapsed if elapsed > 0 else 0

            legend_str = genome_legend(top_genome)
            avg_score = sum(scores) / len(scores)
            stability = tracker.stability()
            print(
                f"Gen {gen+1:>4} | Best: {top_score:.3f} | Avg: {avg_score:.3f} | "
                f"Legend: {legend_str[:25]:25} | "
                f"Stability: {stability:.2f} | "
                f"{n_games:,} games | {gps:,.0f}/s | HoF: {len(hall_of_fame)}"
            )
            write_status(
                "Phase 2: GA Evolution",
                f"Gen {gen+1}/{max_generations}",
                progress=(gen + 1) / max_generations,
                gen=gen + 1, max_gen=max_generations,
                extra={
                    "best_score": top_score,
                    "avg_score": avg_score,
                    "stability": stability,
                    "legend": legend_str,
                    "games_per_sec": int(gps),
                    "hof_size": len(hall_of_fame),
                },
            )

            if converged:
                print(f"\n  Converged after {gen+1} generations (stability >= {cfg['convergence_threshold']:.0%})")
                break

            # Breed next generation — same-legend crossover
            survivors = select(population, scores, top_n)
            next_gen  = survivors.copy()
            while len(next_gen) < population_size:
                p1, p2 = random.sample(survivors, 2)
                if genome_legend(p1) == genome_legend(p2):
                    child = crossover(p1, p2, card_pool)
                else:
                    child = mutate(p1, card_pool, mutation_rate * 2)
                child = mutate(child, card_pool, mutation_rate)
                next_gen.append(child)
            population = next_gen

    meta_tracker.report()
    return best_genome, best_score


# ---------------------------------------------------------------------------
# ML training
# ---------------------------------------------------------------------------

def train_or_load_policy(card_pool, cfg=CONFIG):
    path     = cfg["ml_model_path"]
    trainer  = MLAgentTrainer(card_pool)

    if os.path.exists(path) and not os.environ.get("RETRAIN"):
        print(f"  Loading saved ML policy: {path}")
        trainer.load(path)
    else:
        print(f"\n{'='*50}")
        print("  Phase 1: ML Agent Self-Play Training")
        print(f"{'='*50}")
        trainer.train(
            generations=cfg["ml_generations"],
            games_per_gen=cfg["ml_games_per_gen"],
        )
        trainer.save(path)

    return trainer.policy


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train_or_load_rl(card_pool, cfg=CONFIG):
    """Train or load the deep RL agent (PPO self-play)."""
    path = cfg["rl_model_path"]
    trainer = SelfPlayTrainer(card_pool)

    if os.path.exists(path) and not os.environ.get("RETRAIN"):
        print(f"  Loading saved RL model: {path}")
        trainer.load(path)
    else:
        print(f"\n{'='*50}")
        print("  Phase 1b: Deep RL Self-Play Training (PPO)")
        print(f"{'='*50}")
        trainer.train(
            generations=cfg["rl_generations"],
            games_per_gen=cfg["rl_games_per_gen"],
        )
        trainer.save(path)

    return trainer.net


def main():
    print(f"{'='*50}")
    print("  RIFTBOUND META DECK FINDER")
    print(f"{'='*50}")

    clear_status()
    write_status("Initializing", "Loading card pool and meta decks...")

    card_pool = load_card_pool()
    print(f"  Card pool: {len(card_pool)} cards")

    # Load meta baselines
    print("\n  Loading tournament meta decks...")
    meta_genomes = load_meta_decks("data/meta_decks.json", card_pool)
    if not meta_genomes:
        print("  (none found — add data/meta_decks.json to include tournament baselines)")

    # Phase 1a: Basic ML policy
    write_status("Phase 1a: ML Training", "Training basic ML policy via REINFORCE")
    ml_policy = train_or_load_policy(card_pool)

    # Phase 1b: Deep RL agent
    write_status("Phase 1b: RL Training", "Training deep RL agent via PPO self-play")
    rl_net = train_or_load_rl(card_pool)

    # Benchmark
    write_status("Benchmarking", "Testing RL agent vs Expert heuristic")
    print(f"\n  RL Agent vs Expert Heuristic:")
    wr = benchmark_vs_expert(rl_net, card_pool, games=200)
    print(f"  Win rate: {wr:.0%}")

    # Evolve
    write_status("Phase 2: GA Evolution", "Starting massive parallel evolution")
    print(f"\n{'='*50}")
    print("  Phase 2: Massive Parallel Evolution")
    print(f"{'='*50}")
    t_start = time.time()

    best_genome, best_score = evolve_massive(
        card_pool,
        ml_policy=ml_policy,
        meta_genomes=meta_genomes,
    )

    total_time = time.time() - t_start

    # Final result
    print(f"\n{'='*50}")
    print("  ABSOLUTE BEST DECK")
    print(f"{'='*50}")
    print(summarize_deck(best_genome))
    print(f"\n  Training score : {best_score:.3f}")
    print(f"  Total time     : {total_time/60:.1f} min")

    write_status("Complete", f"Best score: {best_score:.3f}", progress=1.0,
                 extra={"best_score": best_score, "legend": genome_legend(best_genome),
                        "total_time_min": round(total_time / 60, 1)})

    # Save result
    os.makedirs("results", exist_ok=True)
    cards = genome_cards(best_genome)
    with open("results/best_deck.json", "w") as f:
        json.dump({
            "legend": genome_legend(best_genome),
            "score": best_score,
            "deck": list(Counter(cards).items())
        }, f, indent=2)
    print("  Saved to results/best_deck.json")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
