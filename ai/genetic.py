import random
import multiprocessing
from collections import Counter

from game.deck import Deck
from game.player import Player
from game.engine import GameEngine
from game.legend import Legend, load_legends
from game.strategy import ExpertStrategy

_expert = ExpertStrategy()


# ---------------------------------------------------------------------------
# Parallel island worker
# ---------------------------------------------------------------------------

def _island_worker(args):
    """Worker function for parallel island evolution."""
    (legend_name, card_pool, deck_size, island_pop, island_gens, island_top_n,
     mutation_rate, opponent_pool_size, games_per_opponent,
     hall_of_fame_size, coevo_ratio) = args

    try:
        legend_obj = get_legend(legend_name)
        legal_count = len([c for c in card_pool if legend_obj.is_legal(c)])
        if legal_count < deck_size:
            return None

        best_genome, best_score = evolve_island(
            legend=legend_obj, card_pool=card_pool,
            deck_size=deck_size, population_size=island_pop,
            generations=island_gens, top_n=island_top_n,
            mutation_rate=mutation_rate,
            opponent_pool_size=opponent_pool_size,
            games_per_opponent=games_per_opponent,
            hall_of_fame_size=hall_of_fame_size,
            coevo_ratio=coevo_ratio,
            verbose=False,
        )
        return (legend_name, best_genome, best_score)
    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# Genome format: (legend_name, [card_name, ...])
# A legal genome only contains cards from the legend's two domains,
# and must include at least 1 champion matching the legend.
# ---------------------------------------------------------------------------

_legends_cache = None
_legends_by_name = None


def _load_legends():
    global _legends_cache, _legends_by_name
    if _legends_cache is None:
        _legends_cache = load_legends()
        _legends_by_name = {l.name: l for l in _legends_cache}
    return _legends_cache, _legends_by_name


def get_legend(name) -> Legend:
    _, by_name = _load_legends()
    return by_name[name]


def all_legends() -> list:
    legends, _ = _load_legends()
    return legends


# ---------------------------------------------------------------------------
# Genome helpers
# ---------------------------------------------------------------------------

def genome_to_deck(genome, card_pool):
    """Convert a genome (legend_name, card_list) into a Deck."""
    _, cards = genome
    pool_lookup = {card.name: card for card in card_pool}
    return Deck([pool_lookup[name].copy() for name in cards])


def genome_legend(genome) -> str:
    """Return the legend name from a genome."""
    return genome[0]


def genome_cards(genome) -> list:
    """Return the card list from a genome."""
    return genome[1]


def random_genome(card_pool, deck_size=40, legend=None):
    """
    Build a random domain-legal deck for a given legend.
    If no legend specified, picks one at random.
    Guarantees at least 1 champion matching the legend.
    Uses legend identity to weight cards toward the legend's game plan.
    """
    from game.legend_identity import apply_legend_weights, enforce_deck_composition

    legends, by_name = _load_legends()

    if legend is None:
        legend = random.choice(legends)
    elif isinstance(legend, str):
        legend = by_name[legend]

    # Get legal card pools — all domain-legal cards (including other champions)
    all_legal = [c for c in card_pool if legend.is_legal(c)]
    own_champions = legend.get_own_champions(card_pool)

    genome = []
    counts = Counter()

    # Guarantee 1-2 copies of the legend's own champion
    if own_champions:
        champ = random.choice(own_champions)
        n_copies = random.randint(1, min(2, champ.max_copies))
        for _ in range(n_copies):
            genome.append(champ.name)
            counts[champ.name] += 1

    # Fill the rest — weights adjusted by legend identity
    while len(genome) < deck_size:
        available = [c for c in all_legal if counts[c.name] < c.max_copies]
        if not available:
            break
        weights = [c.weight * apply_legend_weights(c, legend.name) for c in available]
        card = random.choices(available, weights=weights, k=1)[0]
        genome.append(card.name)
        counts[card.name] += 1

    # Enforce minimum spell/gear counts for the legend's archetype
    genome = enforce_deck_composition(genome, legend.name, card_pool)

    return (legend.name, genome)


def _legend_domains(genome):
    """Get the two domains from a genome's legend."""
    legend_name = genome_legend(genome)
    try:
        legend = get_legend(legend_name)
        domains = sorted(legend.domains)
        return domains[0], domains[1] if len(domains) > 1 else domains[0]
    except (KeyError, IndexError):
        return "Order", "Order"


def _make_player(name, genome, card_pool, strategy=None):
    """Build a Player with correct dual-domain rune pool from the genome's legend."""
    deck = genome_to_deck(genome, card_pool)
    d1, d2 = _legend_domains(genome)
    return Player(name, deck, domain=d1, domain2=d2, strategy=strategy or _expert)


def _make_opponent(genome, card_pool, ml_policy=None):
    """Build an opponent Player (or MLPlayer) from a genome."""
    deck = genome_to_deck(genome, card_pool)
    d1, d2 = _legend_domains(genome)
    if ml_policy is not None:
        from ai.ml_agent import MLPlayer
        return MLPlayer("MLOpponent", deck, ml_policy, training=False)
    return Player("Opponent", deck, domain=d1, domain2=d2, strategy=_expert)


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def fitness_vs_pool(genome, opponent_genomes, card_pool, ml_policy=None):
    """Win rate of this genome against a mixed opponent pool."""
    wins = 0
    for entry in opponent_genomes:
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], bool):
            opp_genome, use_ml = entry
        else:
            opp_genome, use_ml = entry, False

        p1 = _make_player("Evolved", genome, card_pool)
        p2 = _make_opponent(opp_genome, card_pool, ml_policy if use_ml else None)

        result = GameEngine(p1, p2).play_game()
        if result == 1:
            wins += 1

    return wins / len(opponent_genomes)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_best(genome, card_pool, games=100):
    """Test the best deck against fresh random opponents."""
    wins = 0
    for _ in range(games):
        opp = random_genome(card_pool, len(genome_cards(genome)))
        p1 = _make_player("Best", genome, card_pool)
        p2 = _make_player("Random", opp, card_pool)

        result = GameEngine(p1, p2).play_game()
        if result == 1:
            wins += 1

    return wins / games


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select(population, scores, top_n):
    """Return the top_n genomes ranked by score."""
    ranked = sorted(zip(scores, population), reverse=True)
    return [genome for _, genome in ranked[:top_n]]


# ---------------------------------------------------------------------------
# Crossover (same-legend only)
# ---------------------------------------------------------------------------

def crossover(parent1, parent2, card_pool):
    """
    Combine two genomes. Both must share the same legend.
    Splits card lists at a random point, fixes max_copies violations,
    fills gaps with legal random cards.
    """
    legend_name = genome_legend(parent1)
    legend = get_legend(legend_name)

    cards1 = genome_cards(parent1)
    cards2 = genome_cards(parent2)

    split = random.randint(1, len(cards1) - 1)
    child_cards = cards1[:split] + cards2[split:]

    counts = Counter(child_cards)
    pool_lookup = {card.name: card for card in card_pool}
    fixed = []

    for name in child_cards:
        card = pool_lookup[name]
        if counts[name] > card.max_copies:
            counts[name] -= 1
        else:
            fixed.append(name)

    # Fill missing slots with domain-legal cards
    legal = [c for c in card_pool if legend.is_legal(c)]
    while len(fixed) < len(cards1):
        available = [c for c in legal if counts[c.name] < c.max_copies]
        if not available:
            break
        pick = random.choices(available, weights=[c.weight for c in available], k=1)[0]
        fixed.append(pick.name)
        counts[pick.name] += 1

    return (legend_name, fixed)


# ---------------------------------------------------------------------------
# Mutation (domain-legal only)
# ---------------------------------------------------------------------------

def mutate(genome, card_pool, rate=0.1):
    """Randomly replace cards with legal alternatives, weighted by legend identity."""
    from game.legend_identity import apply_legend_weights, enforce_deck_composition

    legend_name = genome_legend(genome)
    legend = get_legend(legend_name)
    cards = genome_cards(genome)

    legal = [c for c in card_pool if legend.is_legal(c)]

    counts = Counter(cards)
    result = []

    for name in cards:
        if random.random() < rate:
            counts[name] -= 1
            available = [c for c in legal if counts[c.name] < c.max_copies]
            if available:
                weights = [c.weight * apply_legend_weights(c, legend_name) for c in available]
                pick = random.choices(available, weights=weights, k=1)[0]
                result.append(pick.name)
                counts[pick.name] += 1
            else:
                result.append(name)
                counts[name] += 1
        else:
            result.append(name)

    # Enforce composition after mutation
    result = enforce_deck_composition(result, legend_name, card_pool)

    return (legend_name, result)


# ---------------------------------------------------------------------------
# Hall of Fame
# ---------------------------------------------------------------------------

def update_hall_of_fame(hof, genome, max_size):
    """Add genome to hall of fame if not already present."""
    frozen = (genome_legend(genome), tuple(sorted(genome_cards(genome))))
    if any((genome_legend(g), tuple(sorted(genome_cards(g)))) == frozen for g in hof):
        return
    hof.append(genome)
    if len(hof) > max_size:
        hof.pop(0)


# ---------------------------------------------------------------------------
# Main Evolution Loop
# ---------------------------------------------------------------------------

def evolve(
    card_pool,
    deck_size=40,
    population_size=30,
    generations=20,
    top_n=15,
    mutation_rate=0.1,
    opponent_pool_size=12,
    games_per_opponent=10,
    hall_of_fame_size=10,
    coevo_ratio=0.5,
    ml_policy=None,
    ml_ratio=0.0,
    legend=None,
    history=None,
    verbose=True
):
    """
    Evolves domain-legal decks against a mixed opponent pool.

    legend: if set, all decks use this legend. If None, each genome
            picks a random legend — cross-legend crossover not allowed,
            so same-legend pairs are selected for breeding.
    """
    # Seed population from historical winners if available
    seeds = []
    if history:
        from ai.memory import seed_population_from_history
        seeds = seed_population_from_history(history, card_pool, legend, count=min(5, population_size // 4))
        if seeds and verbose:
            print(f"    Seeded {len(seeds)} genomes from history")

    population = list(seeds) + [random_genome(card_pool, deck_size, legend) for _ in range(population_size - len(seeds))]
    hall_of_fame = []

    best_genome = None
    best_score = -1

    if ml_policy is None:
        ml_ratio = 0.0

    for gen in range(generations):
        # Build opponent pool
        n_ml     = int(opponent_pool_size * ml_ratio)
        n_hof    = int(opponent_pool_size * coevo_ratio) if hall_of_fame else 0
        n_random = opponent_pool_size - n_hof - n_ml

        random_opponents = [(random_genome(card_pool, deck_size), False) for _ in range(max(n_random, 0))]
        hof_opponents    = [(random.choice(hall_of_fame), False) for _ in range(n_hof)] if hall_of_fame and n_hof > 0 else []
        ml_opponents     = [(random_genome(card_pool, deck_size), True)  for _ in range(n_ml)]

        opponent_pool = random_opponents + hof_opponents + ml_opponents
        expanded_opponents = opponent_pool * games_per_opponent

        scores = [fitness_vs_pool(g, expanded_opponents, card_pool, ml_policy) for g in population]

        top_score = max(scores)
        top_genome = population[scores.index(top_score)]

        if top_score > best_score:
            best_score = top_score
            best_genome = top_genome

        update_hall_of_fame(hall_of_fame, top_genome, hall_of_fame_size)

        if verbose:
            legend_str = genome_legend(top_genome)
            print(
                f"Generation {gen + 1:>3} | Best: {top_score:.2f} | "
                f"Avg: {sum(scores)/len(scores):.2f} | "
                f"Legend: {legend_str} | HoF: {len(hall_of_fame)}"
            )

        # Breed next generation — same-legend crossover
        survivors = select(population, scores, top_n)
        next_gen = survivors.copy()

        while len(next_gen) < population_size:
            p1, p2 = random.sample(survivors, 2)
            # Only cross if same legend; otherwise mutate p1
            if genome_legend(p1) == genome_legend(p2):
                child = crossover(p1, p2, card_pool)
            else:
                child = mutate(p1, card_pool, mutation_rate * 2)
            child = mutate(child, card_pool, mutation_rate)
            next_gen.append(child)

        population = next_gen

    return best_genome, best_score


def run_genetic_algorithm(
    card_pool,
    population_size=30,
    deck_size=40,
    generations=20,
    keep_top=10,
    mutation_rate=0.1,
    opponent_pool_size=12,
    games_per_opponent=3,
    hall_of_fame_size=10,
    coevo_ratio=0.5,
    ml_policy=None,
    ml_ratio=0.0,
    legend=None,
    history=None,
    verbose=True
):
    """Entry point matching main.py's call signature. Wraps evolve()."""
    return evolve(
        card_pool=card_pool,
        deck_size=deck_size,
        population_size=population_size,
        generations=generations,
        top_n=keep_top,
        mutation_rate=mutation_rate,
        opponent_pool_size=opponent_pool_size,
        games_per_opponent=games_per_opponent,
        hall_of_fame_size=hall_of_fame_size,
        coevo_ratio=coevo_ratio,
        ml_policy=ml_policy,
        ml_ratio=ml_ratio,
        legend=legend,
        history=history,
        verbose=verbose
    )


# ---------------------------------------------------------------------------
# Island Model Evolution
# ---------------------------------------------------------------------------

def evolve_island(legend, card_pool, deck_size=40, population_size=20,
                  generations=30, top_n=10, mutation_rate=0.1,
                  opponent_pool_size=8, games_per_opponent=5,
                  hall_of_fame_size=5, coevo_ratio=0.3,
                  ml_policy=None, ml_ratio=0.0, history=None, verbose=False):
    """
    Run a focused evolution for a single legend.
    Returns (best_genome, best_score).
    """
    return evolve(
        card_pool=card_pool, deck_size=deck_size,
        population_size=population_size, generations=generations,
        top_n=top_n, mutation_rate=mutation_rate,
        opponent_pool_size=opponent_pool_size,
        games_per_opponent=games_per_opponent,
        hall_of_fame_size=hall_of_fame_size,
        coevo_ratio=coevo_ratio, ml_policy=ml_policy,
        ml_ratio=ml_ratio, legend=legend, history=history, verbose=verbose,
    )


def island_tournament(champions, card_pool, games_per_matchup=50):
    """
    Round-robin tournament between island champions.
    Returns list of (genome, win_rate, wins, losses) sorted by win rate.
    """
    n = len(champions)
    records = {i: {"wins": 0, "losses": 0} for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            wr = head_to_head(champions[i], champions[j], card_pool, games=games_per_matchup)
            wins_i = int(wr * games_per_matchup)
            wins_j = games_per_matchup - wins_i

            records[i]["wins"] += wins_i
            records[i]["losses"] += wins_j
            records[j]["wins"] += wins_j
            records[j]["losses"] += wins_i

    results = []
    for i in range(n):
        total = records[i]["wins"] + records[i]["losses"]
        wr = records[i]["wins"] / max(total, 1)
        results.append((champions[i], wr, records[i]["wins"], records[i]["losses"]))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def evolve_islands(
    card_pool,
    legends=None,
    deck_size=40,
    island_pop=20,
    island_gens=30,
    island_top_n=10,
    mutation_rate=0.1,
    opponent_pool_size=8,
    games_per_opponent=5,
    hall_of_fame_size=5,
    coevo_ratio=0.3,
    ml_policy=None,
    ml_ratio=0.0,
    tournament_games=50,
    on_island_complete=None,
    history=None,
    num_workers=None,
    verbose=True,
):
    """
    Island model: run ALL legends in parallel, then tournament.

    1. Each legend gets its own population and evolves independently
       across all available CPU cores simultaneously
    2. The best deck from each island enters a round-robin tournament
    3. The overall winner is the best deck across all legends

    Returns (overall_best_genome, tournament_results).
    """
    if legends is None:
        legends = all_legends()

    num_workers = num_workers or multiprocessing.cpu_count()

    if verbose:
        print(f"\n  Island Model: {len(legends)} legends x {island_gens} gens x {island_pop} pop")
        print(f"  Parallel workers: {num_workers}")

    # Build worker args for each legend
    worker_args = []
    for legend in legends:
        legend_obj = legend if isinstance(legend, Legend) else get_legend(legend)
        worker_args.append((
            legend_obj.name, card_pool, deck_size, island_pop, island_gens,
            island_top_n, mutation_rate, opponent_pool_size, games_per_opponent,
            hall_of_fame_size, coevo_ratio,
        ))

    # Run all islands in parallel
    if verbose:
        print(f"  Launching {len(worker_args)} islands in parallel...")

    with multiprocessing.Pool(processes=min(num_workers, len(worker_args))) as pool:
        results_raw = pool.map(_island_worker, worker_args)

    # Collect results
    champions = []
    champion_details = []

    for i, result in enumerate(results_raw):
        if result is None:
            if verbose:
                legend_name = worker_args[i][0]
                print(f"  {legend_name}: skipped or failed")
            continue

        legend_name, best_genome, best_score = result
        champions.append(best_genome)
        champion_details.append({
            "legend": legend_name,
            "genome": best_genome,
            "island_score": best_score,
        })

        if verbose:
            print(f"  {legend_name:<35} score: {best_score:.3f}")

        if on_island_complete:
            on_island_complete(legend_name, best_genome, best_score, i, len(legends))

    if not champions:
        return None, []

    # Final tournament
    if verbose:
        print(f"\n  Final Tournament: {len(champions)} island champions, {tournament_games} games per matchup")

    results = island_tournament(champions, card_pool, games_per_matchup=tournament_games)

    if verbose:
        print(f"\n  {'Legend':<35} {'Island':>7} {'Tourn':>7} {'W':>4} {'L':>4}")
        print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*4} {'-'*4}")
        for genome, wr, wins, losses in results:
            legend_name = genome_legend(genome)
            island_score = next(
                (d["island_score"] for d in champion_details if d["legend"] == legend_name),
                0,
            )
            print(f"  {legend_name:<35} {island_score:>7.3f} {wr:>7.3f} {wins:>4} {losses:>4}")

    overall_best = results[0][0]
    return overall_best, results


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def summarize_deck(genome):
    """Return a readable summary of a genome."""
    legend_name = genome_legend(genome)
    cards = genome_cards(genome)
    counts = Counter(cards)
    lines = [f"  Legend: {legend_name}"]
    lines += [f"  {count}x {name}" for name, count in sorted(counts.items())]
    return "\n".join(lines)


def head_to_head(genome_a, genome_b, card_pool, games=100):
    """Play two genomes directly. Returns win rate for genome_a."""
    wins = 0
    for _ in range(games):
        p1 = _make_player("A", genome_a, card_pool)
        p2 = _make_player("B", genome_b, card_pool)
        result = GameEngine(p1, p2).play_game()
        if result == 1:
            wins += 1
    return wins / games
