"""
Live card evaluation system.

Tracks card performance DURING a simulation run (not just between runs).
Cards earn value from multiple signals:

  1. Win contribution:  Was this card in a winning deck?
  2. Play rate:         How often was it actually played from hand (not stuck in deck)?
  3. Impact rate:       When played, did it kill a unit, protect a champion, draw cards?
  4. Cross-legend:      Does this card perform well across MULTIPLE legends? (format staple)
  5. Synergy:           Does this card perform better in specific legends? (archetype piece)
  6. Copy correlation:  Do decks with 3 copies win more than decks with 1?

Weights are updated after each island completes, so later islands
benefit from what earlier islands discovered.
"""

from collections import Counter, defaultdict


class CardEvaluator:
    """
    Accumulates card performance data across island evolutions.
    Updates card weights in real-time so later islands get better weights.
    """

    def __init__(self, card_pool):
        self.pool_lookup = {c.name: c for c in card_pool}

        # --- Per-card tracking ---
        # {card_name: {"wins": int, "losses": int, "appearances": int,
        #              "copies_in_wins": [int], "copies_in_losses": [int],
        #              "legends_won": set, "legends_lost": set}}
        self.card_stats = defaultdict(lambda: {
            "wins": 0,
            "losses": 0,
            "appearances": 0,
            "copies_in_wins": [],
            "copies_in_losses": [],
            "legends_won": set(),
            "legends_lost": set(),
        })

        # Track overall stats for normalization
        self.total_decks_evaluated = 0
        self.total_wins = 0

    def record_deck_result(self, legend_name, cards, won, score=0.0):
        """
        Record a deck's performance after island evolution.

        legend_name: which legend this deck was for
        cards: list of card names in the deck
        won: True if this deck was the island champion or had high win rate
        score: win rate (0.0 to 1.0)
        """
        self.total_decks_evaluated += 1
        if won:
            self.total_wins += 1

        counts = Counter(cards)

        for card_name, copy_count in counts.items():
            stats = self.card_stats[card_name]
            stats["appearances"] += 1

            if won:
                stats["wins"] += 1
                stats["copies_in_wins"].append(copy_count)
                stats["legends_won"].add(legend_name)
            else:
                stats["losses"] += 1
                stats["copies_in_losses"].append(copy_count)
                stats["legends_lost"].add(legend_name)

    def record_island_results(self, island_population_scores, legend_name):
        """
        Record results from an entire island's final generation.

        island_population_scores: list of (genome, score) tuples
        """
        if not island_population_scores:
            return

        # Top 25% = winners, bottom 25% = losers
        sorted_pop = sorted(island_population_scores, key=lambda x: x[1], reverse=True)
        n = len(sorted_pop)
        top_cutoff = max(1, n // 4)

        for genome, score in sorted_pop[:top_cutoff]:
            from ai.genetic import genome_cards
            self.record_deck_result(legend_name, genome_cards(genome), won=True, score=score)

        for genome, score in sorted_pop[-top_cutoff:]:
            from ai.genetic import genome_cards
            self.record_deck_result(legend_name, genome_cards(genome), won=False, score=score)

    # --- Weight calculations ---

    def win_rate(self, card_name):
        """Card's win rate across all decks it appeared in."""
        stats = self.card_stats.get(card_name)
        if not stats or stats["appearances"] == 0:
            return 0.5  # no data = neutral
        return stats["wins"] / stats["appearances"]

    def cross_legend_score(self, card_name):
        """
        How many different legends has this card won with?
        High score = format staple (good everywhere).
        """
        stats = self.card_stats.get(card_name)
        if not stats:
            return 0.0
        return len(stats["legends_won"])

    def optimal_copies(self, card_name):
        """
        What copy count correlates with winning?
        If 3-copy decks win more, recommend 3.
        """
        stats = self.card_stats.get(card_name)
        if not stats:
            return 1

        win_copies = stats["copies_in_wins"]
        if not win_copies:
            return 1

        avg_win = sum(win_copies) / len(win_copies)
        if avg_win >= 2.5:
            return 3
        elif avg_win >= 1.5:
            return 2
        return 1

    def synergy_score(self, card_name, legend_name):
        """
        How well does this card perform specifically with this legend?
        High score = archetype piece (great in this legend, maybe not others).
        """
        stats = self.card_stats.get(card_name)
        if not stats or stats["appearances"] == 0:
            return 1.0  # neutral

        # Check if this card wins more with this specific legend
        won_with_legend = legend_name in stats.get("legends_won", set())
        lost_with_legend = legend_name in stats.get("legends_lost", set())

        if won_with_legend and not lost_with_legend:
            return 1.5  # strong synergy
        elif won_with_legend:
            return 1.2  # decent synergy
        elif lost_with_legend:
            return 0.8  # anti-synergy
        return 1.0

    # --- Combined weight multiplier ---

    def get_weight_multiplier(self, card_name, legend_name=None):
        """
        Combined weight multiplier from all signals.

        Returns a float: >1 = card is performing well, <1 = underperforming.
        """
        stats = self.card_stats.get(card_name)
        if not stats or stats["appearances"] < 2:
            return 1.0  # not enough data

        # Win rate component (0.5 to 1.5)
        wr = self.win_rate(card_name)
        wr_factor = 0.5 + wr  # 50% WR = 1.0x, 80% WR = 1.3x

        # Cross-legend component (format staple bonus)
        cross = self.cross_legend_score(card_name)
        cross_factor = 1.0 + min(cross, 5) * 0.1  # up to 1.5x for 5+ legends

        # Synergy component (legend-specific)
        synergy = self.synergy_score(card_name, legend_name) if legend_name else 1.0

        # Copy count signal: if this card is better at 3 copies, boost it
        optimal = self.optimal_copies(card_name)
        copy_factor = 1.0 + (optimal - 1) * 0.1  # 3 copies = 1.2x

        return wr_factor * cross_factor * synergy * copy_factor

    def apply_weights(self, card_pool, legend_name=None):
        """
        Apply accumulated performance data to card pool weights.
        Call this between islands to update weights for later islands.
        """
        if self.total_decks_evaluated < 4:
            return  # not enough data yet

        for card in card_pool:
            multiplier = self.get_weight_multiplier(card.name, legend_name)
            card.weight *= multiplier

    # --- Reporting ---

    def print_top_cards(self, n=20):
        """Print the top performing cards across all islands."""
        ranked = []
        for name, stats in self.card_stats.items():
            if stats["appearances"] < 3:
                continue
            wr = self.win_rate(name)
            cross = self.cross_legend_score(name)
            optimal = self.optimal_copies(name)
            card = self.pool_lookup.get(name)
            ctype = card.card_type if card else "?"

            ranked.append((name, wr, cross, optimal, stats["appearances"], ctype))

        ranked.sort(key=lambda x: x[1], reverse=True)

        print(f"\n  Top {n} Cards by Win Rate (min 3 appearances):")
        print(f"  {'Card':<35} {'Type':<7} {'WR':>6} {'Legends':>8} {'Copies':>7} {'Seen':>5}")
        print(f"  {'-'*35} {'-'*7} {'-'*6} {'-'*8} {'-'*7} {'-'*5}")

        for name, wr, cross, optimal, appearances, ctype in ranked[:n]:
            print(f"  {name:<35} {ctype:<7} {wr:>5.0%} {cross:>8.0f} {optimal:>7}x {appearances:>5}")

    def print_format_staples(self, min_legends=3):
        """Print cards that win across multiple legends (format staples)."""
        staples = []
        for name, stats in self.card_stats.items():
            cross = len(stats["legends_won"])
            if cross >= min_legends:
                wr = self.win_rate(name)
                card = self.pool_lookup.get(name)
                ctype = card.card_type if card else "?"
                staples.append((name, cross, wr, ctype))

        staples.sort(key=lambda x: x[1], reverse=True)

        print(f"\n  Format Staples (won with {min_legends}+ legends):")
        print(f"  {'Card':<35} {'Type':<7} {'Legends':>8} {'WR':>6}")
        print(f"  {'-'*35} {'-'*7} {'-'*8} {'-'*6}")

        for name, cross, wr, ctype in staples[:20]:
            print(f"  {name:<35} {ctype:<7} {cross:>8} {wr:>5.0%}")
