"""
Persistent memory system for the Riftbound simulator.

Accumulates knowledge across simulation runs:
  1. Past winning decks (seed future populations)
  2. Card reputation scores (win-correlation per card)
  3. Legend performance history (win rates across runs)
  4. Matchup matrix (legend vs legend win rates)
  5. Archetype templates (clusters of similar winning decks)

Stored in results/history.json — grows after every run.
"""

import json
import os
from collections import Counter, defaultdict

HISTORY_PATH = "results/history.json"


# ---------------------------------------------------------------------------
# History data structure
# ---------------------------------------------------------------------------

def _empty_history():
    return {
        "runs": [],                    # list of run summaries
        "card_reputation": {},         # card_name -> {"wins": N, "appearances": N}
        "legend_performance": {},      # legend_name -> {"wins": N, "runs": N, "avg_score": F}
        "matchup_matrix": {},          # "legend_a::legend_b" -> {"a_wins": N, "b_wins": N}
        "archetypes": [],              # list of {"name": str, "legend": str, "core_cards": [...], "win_rate": F}
        "top_decks": [],               # list of past winning genomes for seeding
    }


def load_history() -> dict:
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    return _empty_history()


def save_history(history: dict):
    os.makedirs(os.path.dirname(HISTORY_PATH) or "results", exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# 1. Record a completed run
# ---------------------------------------------------------------------------

def record_run(history, tournament_results, card_pool_lookup):
    """
    Record results from a completed simulation run.

    tournament_results: list of (genome, win_rate, wins, losses)
    card_pool_lookup: {card_name: Card} for enrichment
    """
    from ai.genetic import genome_legend, genome_cards

    run_summary = {
        "standings": [],
        "timestamp": _timestamp(),
    }

    for genome, wr, wins, losses in tournament_results:
        legend = genome_legend(genome)
        cards = genome_cards(genome)

        run_summary["standings"].append({
            "legend": legend,
            "win_rate": wr,
            "wins": wins,
            "losses": losses,
            "cards": list(Counter(cards).items()),
        })

        # Update card reputation
        _update_card_reputation(history, cards, wr)

        # Update legend performance
        _update_legend_performance(history, legend, wr)

    # Update matchup matrix from tournament
    _update_matchup_matrix(history, tournament_results)

    # Update archetypes
    _update_archetypes(history, tournament_results, card_pool_lookup)

    # Store top decks for future seeding (keep top 3 per run, max 30 total)
    for genome, wr, _, _ in tournament_results[:3]:
        legend = genome_legend(genome)
        cards = genome_cards(genome)
        history["top_decks"].append({
            "legend": legend,
            "cards": cards,
            "win_rate": wr,
            "timestamp": _timestamp(),
        })
    # Cap at 30 stored decks (keep most recent/highest win rate)
    history["top_decks"] = sorted(
        history["top_decks"],
        key=lambda d: d["win_rate"],
        reverse=True
    )[:30]

    history["runs"].append(run_summary)
    save_history(history)


# ---------------------------------------------------------------------------
# 2. Card reputation scores
# ---------------------------------------------------------------------------

def _update_card_reputation(history, cards, win_rate):
    """Track how often each card appears in winning decks and at what win rate."""
    rep = history["card_reputation"]
    counts = Counter(cards)

    for card_name, count in counts.items():
        if card_name not in rep:
            rep[card_name] = {"wins": 0.0, "appearances": 0, "total_copies": 0}

        rep[card_name]["appearances"] += 1
        rep[card_name]["total_copies"] += count
        rep[card_name]["wins"] += win_rate  # weighted by win rate


def get_card_reputation_weights(history, card_pool) -> dict:
    """
    Convert card reputation into weight multipliers for deck building.

    Cards that frequently appear in winning decks get higher weights.
    New/unseen cards get a neutral weight of 1.0.

    Returns: {card_name: weight_multiplier}
    """
    rep = history.get("card_reputation", {})
    if not rep:
        return {}

    weights = {}
    max_appearances = max((r["appearances"] for r in rep.values()), default=1)

    for card_name, data in rep.items():
        appearances = data["appearances"]
        avg_wr = data["wins"] / max(appearances, 1)

        # Score: frequency * win_rate, normalized
        frequency = appearances / max(max_appearances, 1)
        reputation = frequency * 0.4 + avg_wr * 0.6

        # Convert to weight multiplier: 0.5x (bad) to 2.0x (great)
        weights[card_name] = 0.5 + reputation * 1.5

    return weights


def apply_reputation_weights(card_pool, history):
    """Adjust card weights in the pool based on reputation from past runs."""
    rep_weights = get_card_reputation_weights(history, card_pool)
    if not rep_weights:
        return

    for card in card_pool:
        if card.name in rep_weights:
            card.weight *= rep_weights[card.name]


def get_optimal_copies(history, legend_name: str) -> dict:
    """
    Based on past winning decks, determine optimal copy counts for cards.
    Cards that consistently appear at 3 copies in winners should be at 3.
    Cards that never show up should be at 0.

    Returns: {card_name: recommended_copies}
    """
    top_decks = history.get("top_decks", [])
    card_copy_counts = {}  # card_name -> list of copy counts across decks

    for deck in top_decks:
        if deck.get("legend") != legend_name:
            continue

        counts = {}
        for card_name in deck.get("cards", []):
            counts[card_name] = counts.get(card_name, 0) + 1

        for card_name, count in counts.items():
            if card_name not in card_copy_counts:
                card_copy_counts[card_name] = []
            card_copy_counts[card_name].append(count)

    # Average copy count across winning decks
    recommendations = {}
    for card_name, copy_list in card_copy_counts.items():
        avg = sum(copy_list) / len(copy_list)
        if avg >= 2.5:
            recommendations[card_name] = 3
        elif avg >= 1.5:
            recommendations[card_name] = 2
        elif avg >= 0.5:
            recommendations[card_name] = 1

    return recommendations


# ---------------------------------------------------------------------------
# 3. Legend performance history
# ---------------------------------------------------------------------------

def _update_legend_performance(history, legend_name, win_rate):
    perf = history["legend_performance"]
    if legend_name not in perf:
        perf[legend_name] = {"wins": 0, "runs": 0, "total_score": 0.0}

    perf[legend_name]["runs"] += 1
    perf[legend_name]["total_score"] += win_rate
    if win_rate > 0.5:
        perf[legend_name]["wins"] += 1


def get_legend_rankings(history) -> list:
    """
    Return legends sorted by historical performance.
    Returns: [(legend_name, avg_score, num_runs), ...]
    """
    perf = history.get("legend_performance", {})
    rankings = []
    for name, data in perf.items():
        avg = data["total_score"] / max(data["runs"], 1)
        rankings.append((name, avg, data["runs"]))

    return sorted(rankings, key=lambda x: x[1], reverse=True)


def get_legend_budget(history, all_legends, total_budget=100):
    """
    Allocate island population/generations budget based on past performance.

    Strong legends get more resources, weak legends get minimum baseline.
    New/untested legends get a fair starting allocation.

    Returns: {legend_name: budget_multiplier}
    """
    perf = history.get("legend_performance", {})
    if not perf:
        return {l.name: 1.0 for l in all_legends}

    budgets = {}
    for legend in all_legends:
        data = perf.get(legend.name)
        if data and data["runs"] >= 2:
            avg = data["total_score"] / data["runs"]
            # Scale: 0.5x for worst, 2.0x for best
            budgets[legend.name] = max(0.5, min(2.0, avg * 2))
        else:
            # Untested = give fair chance
            budgets[legend.name] = 1.0

    return budgets


# ---------------------------------------------------------------------------
# 4. Matchup matrix
# ---------------------------------------------------------------------------

def _update_matchup_matrix(history, tournament_results):
    """Update head-to-head legend matchup data from tournament results."""
    from ai.genetic import genome_legend

    matrix = history["matchup_matrix"]
    legends = [(genome_legend(g), g, wr) for g, wr, _, _ in tournament_results]

    # Infer matchups from relative positioning
    # Higher-ranked legends "beat" lower-ranked ones
    for i, (leg_a, _, wr_a) in enumerate(legends):
        for j, (leg_b, _, wr_b) in enumerate(legends):
            if i >= j:
                continue
            key = f"{leg_a}::{leg_b}"
            rev_key = f"{leg_b}::{leg_a}"

            if key not in matrix:
                matrix[key] = {"a_wins": 0, "b_wins": 0}
            if rev_key not in matrix:
                matrix[rev_key] = {"a_wins": 0, "b_wins": 0}

            # Higher win rate = "won" this matchup
            if wr_a > wr_b:
                matrix[key]["a_wins"] += 1
                matrix[rev_key]["b_wins"] += 1
            elif wr_b > wr_a:
                matrix[key]["b_wins"] += 1
                matrix[rev_key]["a_wins"] += 1


def get_matchup(history, legend_a, legend_b) -> float:
    """
    Get estimated win rate of legend_a vs legend_b.
    Returns 0.5 if no data.
    """
    key = f"{legend_a}::{legend_b}"
    matrix = history.get("matchup_matrix", {})
    data = matrix.get(key)
    if not data:
        return 0.5
    total = data["a_wins"] + data["b_wins"]
    if total == 0:
        return 0.5
    return data["a_wins"] / total


def get_worst_matchups(history, legend_name, top_n=3) -> list:
    """Find a legend's worst matchups. Returns [(opponent, win_rate), ...]."""
    matrix = history.get("matchup_matrix", {})
    matchups = []
    for key, data in matrix.items():
        parts = key.split("::")
        if len(parts) != 2:
            continue
        if parts[0] == legend_name:
            total = data["a_wins"] + data["b_wins"]
            if total > 0:
                wr = data["a_wins"] / total
                matchups.append((parts[1], wr))

    return sorted(matchups, key=lambda x: x[1])[:top_n]


# ---------------------------------------------------------------------------
# 5. Archetype clustering
# ---------------------------------------------------------------------------

def _update_archetypes(history, tournament_results, card_pool_lookup):
    """
    Detect deck archetypes by clustering similar winning decks.

    Two decks are "same archetype" if they share 70%+ of their cards.
    """
    from ai.genetic import genome_legend, genome_cards

    archetypes = history.get("archetypes", [])

    for genome, wr, _, _ in tournament_results:
        if wr < 0.4:
            continue  # only track decks that actually won games

        legend = genome_legend(genome)
        cards = genome_cards(genome)
        card_set = Counter(cards)

        # Try to match to existing archetype
        matched = False
        for arch in archetypes:
            if arch["legend"] != legend:
                continue

            # Calculate card overlap
            arch_cards = Counter(dict(arch["core_cards"]))
            overlap = sum(min(card_set[k], arch_cards[k]) for k in card_set)
            total = sum(card_set.values())
            similarity = overlap / max(total, 1)

            if similarity >= 0.7:
                # Same archetype — update core cards and stats
                arch["appearances"] += 1
                arch["total_win_rate"] += wr
                # Update core cards: keep cards that appear in 60%+ of instances
                _merge_core_cards(arch, card_set)
                matched = True
                break

        if not matched:
            # New archetype
            archetypes.append({
                "legend": legend,
                "name": _auto_name_archetype(legend, cards, card_pool_lookup),
                "core_cards": list(card_set.items()),
                "appearances": 1,
                "total_win_rate": wr,
            })

    # Cap at 50 archetypes
    archetypes = sorted(archetypes, key=lambda a: a["total_win_rate"] / max(a["appearances"], 1), reverse=True)[:50]
    history["archetypes"] = archetypes


def _merge_core_cards(archetype, new_cards):
    """Merge new card counts into archetype core, keeping frequently appearing cards."""
    existing = Counter(dict(archetype["core_cards"]))
    n = archetype["appearances"]

    # Weighted average of card counts
    merged = {}
    all_cards = set(list(existing.keys()) + list(new_cards.keys()))
    for card in all_cards:
        old_count = existing.get(card, 0)
        new_count = new_cards.get(card, 0)
        avg = (old_count * (n - 1) + new_count) / n
        if avg >= 0.5:  # appears in at least half of instances
            merged[card] = round(avg)

    archetype["core_cards"] = list(merged.items())


def _auto_name_archetype(legend, cards, card_pool_lookup):
    """Generate a descriptive name for an archetype based on key cards."""
    legend_short = legend.split(" - ")[0]
    counts = Counter(cards)

    # Find the most distinctive non-staple card
    distinctive = []
    for name, count in counts.most_common():
        card = card_pool_lookup.get(name)
        if card and count >= 2:
            if card.champion:
                distinctive.append(name.split(" - ")[0])
            elif card.cost >= 5:
                distinctive.append(name)

    if distinctive:
        return f"{distinctive[0]} {legend_short}"
    return f"{legend_short} Standard"


def get_archetype_templates(history, legend_name=None) -> list:
    """
    Get archetype templates for seeding GA populations.
    Returns list of (legend, card_list) genomes.
    """
    archetypes = history.get("archetypes", [])
    templates = []

    for arch in archetypes:
        if legend_name and arch["legend"] != legend_name:
            continue
        if arch["appearances"] < 1:
            continue

        cards = []
        for name, count in arch["core_cards"]:
            cards.extend([name] * int(count))

        if len(cards) >= 30:  # need enough core cards for a viable template
            templates.append((arch["legend"], cards, arch["total_win_rate"] / arch["appearances"]))

    return sorted(templates, key=lambda x: x[2], reverse=True)


# ---------------------------------------------------------------------------
# 6. Seed population from history
# ---------------------------------------------------------------------------

def seed_population_from_history(history, card_pool, legend=None, count=5):
    """
    Generate seed genomes from past winners and archetype templates.

    Returns a list of (legend_name, card_list) genomes to inject
    into the starting population.
    """
    from ai.genetic import random_genome, genome_legend, genome_cards
    from collections import Counter

    seeds = []

    # Past winners for this legend (or any legend)
    top_decks = history.get("top_decks", [])
    for deck in top_decks:
        if legend and deck["legend"] != (legend.name if hasattr(legend, 'name') else legend):
            continue
        genome = (deck["legend"], deck["cards"])
        seeds.append(genome)
        if len(seeds) >= count:
            break

    # Archetype templates
    if len(seeds) < count:
        legend_name = legend.name if hasattr(legend, 'name') else legend if legend else None
        templates = get_archetype_templates(history, legend_name)
        for leg, cards, wr in templates:
            if len(seeds) >= count:
                break

            # Pad template to 40 cards if needed
            if len(cards) < 40:
                # Fill remaining slots with random legal cards
                try:
                    from game.legend import Legend
                    full = random_genome(card_pool, 40, leg)
                    # Replace the random cards with template cards
                    template_counts = Counter(cards)
                    full_cards = genome_cards(full)
                    final = list(cards)
                    remaining = [c for c in full_cards if c not in template_counts or template_counts[c] <= 0]
                    while len(final) < 40 and remaining:
                        final.append(remaining.pop(0))
                    seeds.append((leg, final[:40]))
                except Exception:
                    seeds.append((leg, cards))
            else:
                seeds.append((leg, cards[:40]))

    return seeds[:count]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _timestamp():
    import time
    return time.time()


def print_history_summary(history):
    """Print a human-readable summary of accumulated knowledge."""
    runs = len(history.get("runs", []))
    cards_tracked = len(history.get("card_reputation", {}))
    legends_tracked = len(history.get("legend_performance", {}))
    archetypes = len(history.get("archetypes", []))
    top_decks = len(history.get("top_decks", []))

    print(f"\n  Historical Knowledge Base")
    print(f"  Runs completed:    {runs}")
    print(f"  Cards tracked:     {cards_tracked}")
    print(f"  Legends tracked:   {legends_tracked}")
    print(f"  Archetypes found:  {archetypes}")
    print(f"  Stored top decks:  {top_decks}")

    if legends_tracked:
        print(f"\n  Legend Rankings (historical):")
        for name, avg, num_runs in get_legend_rankings(history)[:10]:
            print(f"    {name:<35} avg: {avg:.3f} ({num_runs} runs)")

    if archetypes:
        print(f"\n  Known Archetypes:")
        for arch in history.get("archetypes", [])[:10]:
            avg_wr = arch["total_win_rate"] / max(arch["appearances"], 1)
            print(f"    {arch['name']:<35} {avg_wr:.0%} ({arch['appearances']} appearances)")
