#!/usr/bin/env python3
import pickle
import math
from typing import Dict, List, Tuple

class FastRewardCalculator:
    def __init__(self, cache_file: str, epsilon: float = 1e-9):
        with open(cache_file, "rb") as f:
            cache = pickle.load(f)
        self._tri_probs: Dict[str, float] = cache["trigram_probs"]
        self._eps: float = float(epsilon)
        self.token_lm = _TokenLM(self._tri_probs, self._eps)

        # Precompute expected rewards per (t1, t2)
        self._expected_rewards: Dict[Tuple[str, str], float] = {}
        accum = {}
        for key, p in self._tri_probs.items():
            t1, t2, t3 = key.split(",", 2)
            p = max(p, self._eps)
            accum.setdefault((t1, t2), []).append(-p * math.log(p))
        for bigram, vals in accum.items():
            self._expected_rewards[bigram] = sum(vals) / len(vals)


    def calculate_reward_tokens(self, tokens: List[str], normalize: bool = True) -> float:
        """
        Args:
            tokens (List[str]):
                List of token strings

            normalize (bool, optional):
                Whether to compute the average reward per trigram (True)
                or the unnormalized total reward (False).

        Returns:
            float:
                Returns 0.0 if fewer than 3 tokens are provided.
        """
        if len(tokens) < 3:
            return 0.0
        
        total_reward = 0.0
        num_trigrams = len(tokens) - 2

        for i in range(num_trigrams):
            t1 = tokens[i]
            t2 = tokens[i+1]
            t3 = tokens[i+2]
            
            total_reward -= self.token_lm.logp(t1, t2, t3)

        if normalize:
            return total_reward / num_trigrams
        else:
            return total_reward

class _TokenLM:
    """Minimal token-trigram LM with logp only. Internal use."""
    def __init__(self, tri_probs: Dict[str, float], eps: float):
        self._tri = tri_probs
        self._eps = eps

    @staticmethod
    def _key(t1: str, t2: str, t3: str) -> str:
        return f"{t1},{t2},{t3}"

    def logp(self, t1: str, t2: str, t3: str) -> float:
        """Return log P(t3 | t1, t2) with epsilon floor."""
        p = self._tri.get(self._key(t1, t2, t3), 0.0)
        if p <= 0.0:
            p = self._eps
        return math.log(p)
