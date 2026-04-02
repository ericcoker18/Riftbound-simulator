import os
import statistics
from collections import Counter

from ai.genetic import (
    run_genetic_algorithm, summarize_deck, evaluate_best,
    head_to_head, genome_to_deck, random_genome,
    genome_legend, genome_cards,
)
from ai.ml_agent import MLAgentTrainer, MLPlayer
from game.loader import load_card_pool
from game.engine import GameEngine
from game.player import Player


# ---------------------------------------------------------------------------
# ML training phase
# ---------------------------------------------------------------------------

def train_or_load_policy(card_pool, model_path="models/policy.pt",
                          generations=50, games_per_gen=20):
    trainer = MLAgentTrainer(card_pool)

    if os.path.exists(model_path):
        print(f"\nLoading saved ML policy from {model_path}")
        trainer.load(model_path)
    else:
        print(f"\n{'='*40}")
        print("  Phase 1: ML Agent Self-Play Training")
        print(f"{'='*40}")
        trainer.train(generations=generations, games_per_gen=games_per_gen)
        trainer.save(model_path)

    return trainer.policy


# ---------------------------------------------------------------------------
# GA evolution phase
# ---------------------------------------------------------------------------

def run_named(name, card_pool, ml_policy=None):
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")

    best_deck, best_score = run_genetic_algorithm(
        card_pool=card_pool,
        population_size=30,
        deck_size=40,
        generations=10,
        keep_top=15,
        mutation_rate=0.10,
        opponent_pool_size=12,
        games_per_opponent=10,
        hall_of_fame_size=10,
        coevo_ratio=0.3,
        ml_policy=ml_policy,
        ml_ratio=0.3,
    )

    true_score = evaluate_best(best_deck, card_pool, games=100)

    print(f"\nBest deck ({name}):")
    print(summarize_deck(best_deck))
    print(f"Training score : {best_score:.3f}")
    print(f"True win rate  : {true_score:.3f}")

    return {"name": name, "deck": best_deck, "true_score": true_score}


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def deck_composition(genome, card_pool):
    pool_lookup = {card.name: card for card in card_pool}
    cards = genome_cards(genome)
    counts = Counter(cards)

    total = len(cards)
    total_cost = sum(pool_lookup[n].cost * c for n, c in counts.items())
    total_health = sum(pool_lookup[n].health * c for n, c in counts.items())
    tag_counts = Counter(
        tag for n, c in counts.items()
        for tag in pool_lookup[n].tags
        for _ in range(c)
    )
    by_type = Counter(pool_lookup[n].card_type for n, c in counts.items() for _ in range(c))

    print(f"  Legend          : {genome_legend(genome)}")
    print(f"  Cards           : {total}")
    print(f"  Avg cost        : {total_cost / total:.2f}")
    print(f"  Avg might       : {total_health / total:.2f}")
    print(f"  Types           : {dict(by_type)}")
    if tag_counts:
        tag_str = "  ".join(f"{tag}: {count}" for tag, count in tag_counts.most_common(5))
        print(f"  Top tags        : {tag_str}")


def matchup_results(results, card_pool):
    print(f"\n{'='*40}")
    print("  Matchup Results (head-to-head)")
    print(f"{'='*40}")
    names = [r["name"] for r in results]
    header = f"  {'':12}" + "".join(f"{n:>10}" for n in names)
    print(header)

    for r_a in results:
        row = f"  {r_a['name']:<12}"
        for r_b in results:
            if r_a["name"] == r_b["name"]:
                row += f"{'---':>10}"
            else:
                score = head_to_head(r_a["deck"], r_b["deck"], card_pool, games=100)
                row += f"{score:>10.3f}"
        print(row)


def find_meta_deck(results):
    frozen = [tuple(sorted(genome_cards(r["deck"]))) for r in results]
    for i, deck_a in enumerate(frozen):
        for j, deck_b in enumerate(frozen):
            if i != j and deck_a == deck_b:
                return results[i]
    return None


def compare_runs(results, card_pool):
    scores = [r["true_score"] for r in results]
    avg = statistics.mean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0

    print(f"\n{'='*40}")
    print("  Consistency Report")
    print(f"{'='*40}")

    all_decks = [Counter(genome_cards(r["deck"])) for r in results]
    all_cards = set(name for deck in all_decks for name in deck)

    for card in sorted(all_cards):
        counts = [deck[card] for deck in all_decks]
        if any(c > 0 for c in counts):
            counts_str = "  ".join(f"{r['name']}: {c}x" for r, c in zip(results, counts))
            print(f"  {card:<30} {counts_str}")

    print(f"\n  {'Run':<12} {'Legend':<30} {'Win Rate':>10}")
    for r in results:
        print(f"  {r['name']:<12} {genome_legend(r['deck']):<30} {r['true_score']:>10.3f}")
    print(f"  {'Average':<12} {'':30} {avg:>10.3f}")
    print(f"  {'Std Dev':<12} {'':30} {std:>10.3f}")

    print(f"\n{'='*40}")
    print("  Deck Composition")
    print(f"{'='*40}")
    for r in results:
        print(f"\n  {r['name']}:")
        deck_composition(r["deck"], card_pool)

    matchup_results(results, card_pool)

    meta = find_meta_deck(results)
    if meta:
        print(f"\n{'='*40}")
        print("  *** META DECK DETECTED ***")
        print(f"{'='*40}")
        print(summarize_deck(meta["deck"]))
        print(f"  Win rate: {meta['true_score']:.3f}")
    else:
        print("\n  No identical decks found across runs.")


# ---------------------------------------------------------------------------
# ML benchmark
# ---------------------------------------------------------------------------

def benchmark_vs_ml(results, card_pool, ml_policy, games=100):
    print(f"\n{'='*40}")
    print("  Benchmark vs ML Agent")
    print(f"{'='*40}")

    for r in results:
        wins = 0
        for _ in range(games):
            d1 = genome_to_deck(r["deck"], card_pool)
            d2 = genome_to_deck(random_genome(card_pool, len(genome_cards(r["deck"]))), card_pool)
            p1 = Player(r["name"], d1)
            p2 = MLPlayer("MLAgent", d2, ml_policy, training=False)
            result = GameEngine(p1, p2).play_game()
            if result == 1:
                wins += 1
        print(f"  {r['name']:<12} vs ML: {wins/games:.3f} win rate")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    card_pool = load_card_pool()

    ml_policy = train_or_load_policy(
        card_pool,
        model_path="models/policy.pt",
        generations=50,
        games_per_gen=20,
    )

    print(f"\n{'='*40}")
    print("  Phase 2: Genetic Algorithm Evolution")
    print(f"{'='*40}")

    results = [
        run_named("Run 1", card_pool, ml_policy),
        run_named("Run 2", card_pool, ml_policy),
        run_named("Run 3", card_pool, ml_policy),
    ]

    compare_runs(results, card_pool)
    benchmark_vs_ml(results, card_pool, ml_policy)


if __name__ == "__main__":
    main()
