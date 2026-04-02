import random
from collections import Counter

from game.deck import Deck
from game.player import Player
from game.engine import GameEngine
from game.legend import Legend, load_legends
from game.strategy import ExpertStrategy

_expert = ExpertStrategy()


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
    """
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

    # Fill the rest from all domain-legal cards (units, spells, gear, AND other champions)
    while len(genome) < deck_size:
        available = [c for c in all_legal if counts[c.name] < c.max_copies]
        if not available:
            break
        card = random.choices(available, weights=[c.weight for c in available], k=1)[0]
        genome.append(card.name)
        counts[card.name] += 1

    return (legend.name, genome)


def _make_opponent(genome, card_pool, ml_policy=None):
    """Build an opponent Player (or MLPlayer) from a genome."""
    deck = genome_to_deck(genome, card_pool)
    if ml_policy is not None:
        from ai.ml_agent import MLPlayer
        return MLPlayer("MLOpponent", deck, ml_policy, training=False)
    return Player("Opponent", deck, strategy=_expert)


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

        deck1 = genome_to_deck(genome, card_pool)
        p1 = Player("Evolved", deck1, strategy=_expert)
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
        deck1 = genome_to_deck(genome, card_pool)
        opp = random_genome(card_pool, len(genome_cards(genome)))
        deck2 = genome_to_deck(opp, card_pool)

        p1 = Player("Best", deck1, strategy=_expert)
        p2 = Player("Random", deck2, strategy=_expert)

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
    """Randomly replace cards with legal alternatives."""
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
                pick = random.choices(available, weights=[c.weight for c in available], k=1)[0]
                result.append(pick.name)
                counts[pick.name] += 1
            else:
                result.append(name)
                counts[name] += 1
        else:
            result.append(name)

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
    verbose=True
):
    """
    Evolves domain-legal decks against a mixed opponent pool.

    legend: if set, all decks use this legend. If None, each genome
            picks a random legend — cross-legend crossover not allowed,
            so same-legend pairs are selected for breeding.
    """
    population = [random_genome(card_pool, deck_size, legend) for _ in range(population_size)]
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
        verbose=verbose
    )


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
        deck1 = genome_to_deck(genome_a, card_pool)
        deck2 = genome_to_deck(genome_b, card_pool)
        p1 = Player("A", deck1, strategy=_expert)
        p2 = Player("B", deck2, strategy=_expert)
        result = GameEngine(p1, p2).play_game()
        if result == 1:
            wins += 1
    return wins / games
