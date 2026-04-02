"""
Parallel game evaluation using multiprocessing.

Each worker process loads the card pool once, then evaluates
genome fitness as games come in. No shared state between workers.
"""

import multiprocessing
from ai.genetic import fitness_vs_pool, random_genome

# ---------------------------------------------------------------------------
# Worker process globals (set once per worker via initializer)
# ---------------------------------------------------------------------------

_card_pool = None
_ml_policy = None


def _init_worker(card_pool, ml_policy_state):
    """Called once per worker process to set up shared state."""
    global _card_pool, _ml_policy
    _card_pool = card_pool

    if ml_policy_state is not None:
        import torch
        from ai.ml_agent import PolicyNetwork
        # Workers always use CPU (game sim is CPU-bound, GPU is for training)
        _ml_policy = PolicyNetwork()
        _ml_policy.load_state_dict(ml_policy_state, strict=False)
        _ml_policy.eval()
    else:
        _ml_policy = None


def _eval_genome(args):
    """Worker function: evaluate one genome against the opponent pool."""
    genome, opponent_entries = args
    return fitness_vs_pool(genome, opponent_entries, _card_pool, _ml_policy)


def _eval_genome_batch(args):
    """Worker function: evaluate a batch of genomes. Reduces IPC overhead."""
    genomes, opponent_entries = args
    return [fitness_vs_pool(g, opponent_entries, _card_pool, _ml_policy) for g in genomes]


# ---------------------------------------------------------------------------
# Parallel evaluator
# ---------------------------------------------------------------------------

class ParallelEvaluator:
    """
    Wraps a multiprocessing.Pool to evaluate all genomes in a generation
    simultaneously across all available CPU cores.

    Uses batched tasks (multiple genomes per IPC call) to minimize pickling
    overhead — critical when games are fast.
    """

    def __init__(self, card_pool, ml_policy=None, num_workers=None):
        self.card_pool   = card_pool
        self.num_workers = num_workers or multiprocessing.cpu_count()

        # Serialize ML policy weights (PyTorch state_dict is picklable)
        ml_state = ml_policy.state_dict() if ml_policy is not None else None

        self._pool = multiprocessing.Pool(
            processes=self.num_workers,
            initializer=_init_worker,
            initargs=(card_pool, ml_state),
        )
        print(f"  Parallel evaluator: {self.num_workers} workers")

    def evaluate(self, population, opponent_entries):
        """
        Evaluate all genomes in parallel using batched IPC.
        Each worker gets (pop_size / num_workers) genomes per call.
        Returns list of scores aligned with population order.
        """
        n = len(population)
        batch_size = max(1, n // self.num_workers)

        batches = [
            (population[i:i + batch_size], opponent_entries)
            for i in range(0, n, batch_size)
        ]

        batch_results = self._pool.map(_eval_genome_batch, batches)

        # Flatten batch results back into a flat scores list
        scores = []
        for batch in batch_results:
            scores.extend(batch)
        return scores

    def close(self):
        self._pool.close()
        self._pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
