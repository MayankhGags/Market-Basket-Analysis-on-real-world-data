"""
Apriori Algorithm Implementation.

Implements the classic Apriori algorithm for mining frequent itemsets
from transactional databases using the downward-closure (Apriori) property.

Key Concept:
All non-empty subsets of a frequent itemset must also be frequent.
If an itemset is infrequent, all of its supersets will also be infrequent.
"""

from collections import defaultdict
from itertools import combinations
from typing import Dict, FrozenSet, List, Set, Tuple


class AprioriMiner:
    """Mines frequent itemsets using the Apriori algorithm."""

    def __init__(self, min_support: float):
        """
        Args:
            min_support: Minimum support threshold (float between 0.0 and 1.0).
        """
        if not (0.0 < min_support <= 1.0):
            raise ValueError("min_support must be in range (0.0, 1.0].")
        self.min_support = min_support
        self.total_transactions = 0
        # Stores mapping from frozenset(itemset) -> support float
        self.frequent_itemsets: Dict[FrozenSet[str], float] = {}
        # Stores frequent itemsets grouped by itemset length k: {1: [...], 2: [...]}
        self.itemsets_by_level: Dict[int, List[FrozenSet[str]]] = defaultdict(list)

    def _get_frequent_1_itemsets(
        self, baskets: List[Set[str]]
    ) -> Tuple[Dict[FrozenSet[str], float], Set[str]]:
        """
        Identifies frequent 1-itemsets (L1) by scanning all transactions.

        Args:
            baskets: List of transaction itemsets.

        Returns:
            Tuple of (L1 dictionary {frozenset: support}, set of frequent single items).
        """
        item_counts: Dict[str, int] = defaultdict(int)
        for basket in baskets:
            for item in basket:
                item_counts[item] += 1

        l1: Dict[FrozenSet[str], float] = {}
        frequent_single_items: Set[str] = set()

        for item, count in item_counts.items():
            support = count / self.total_transactions
            if support >= self.min_support:
                itemset = frozenset([item])
                l1[itemset] = support
                frequent_single_items.add(item)

        return l1, frequent_single_items

    def _generate_candidates_ck(
        self, prev_frequent: List[FrozenSet[str]], k: int
    ) -> Set[FrozenSet[str]]:
        """
        Generates candidate k-itemsets (Ck) from frequent (k-1)-itemsets (L_{k-1})
        using the Join and Prune steps.

        Join Step:
        Join pairs of itemsets from L_{k-1} whose union contains exactly k items.

        Prune Step (Apriori Property):
        Verify that all (k-1)-subsets of the candidate are members of L_{k-1}.
        If any subset is missing, the candidate cannot be frequent and is pruned.

        Args:
            prev_frequent: List of frequent (k-1)-itemsets.
            k: The target candidate itemset size.

        Returns:
            Set of candidate k-itemsets that survived pruning.
        """
        prev_set = set(prev_frequent)
        candidates: Set[FrozenSet[str]] = set()

        # Join step
        num_prev = len(prev_frequent)
        for i in range(num_prev):
            for j in range(i + 1, num_prev):
                union_set = prev_frequent[i] | prev_frequent[j]
                if len(union_set) == k:
                    # Prune step: check all (k-1) subsets
                    all_subsets_frequent = True
                    for subset in combinations(union_set, k - 1):
                        if frozenset(subset) not in prev_set:
                            all_subsets_frequent = False
                            break
                    if all_subsets_frequent:
                        candidates.add(union_set)

        return candidates

    def fit(self, baskets: List[Set[str]]) -> Dict[FrozenSet[str], float]:
        """
        Executes the Apriori algorithm on the transaction baskets.

        Args:
            baskets: List of transaction sets.

        Returns:
            Dictionary of all frequent itemsets and their calculated support.
        """
        self.total_transactions = len(baskets)
        if self.total_transactions == 0:
            return {}

        self.frequent_itemsets.clear()
        self.itemsets_by_level.clear()

        # 1. Level 1: Frequent 1-itemsets (L1)
        l1, _ = self._get_frequent_1_itemsets(baskets)
        if not l1:
            return {}

        for itemset, sup in l1.items():
            self.frequent_itemsets[itemset] = sup
            self.itemsets_by_level[1].append(itemset)

        current_frequent = list(l1.keys())
        k = 2

        # 2. Iterate for k >= 2 until no candidates survive
        while current_frequent:
            candidates_k = self._generate_candidates_ck(current_frequent, k)
            if not candidates_k:
                break

            # Count support for candidates in C_k by scanning dataset
            candidate_counts: Dict[FrozenSet[str], int] = defaultdict(int)
            for basket in baskets:
                # Optimized check: only consider candidates subset of basket
                for candidate in candidates_k:
                    if candidate.issubset(basket):
                        candidate_counts[candidate] += 1

            # Filter candidates satisfying min_support -> L_k
            current_frequent = []
            for candidate, count in candidate_counts.items():
                support = count / self.total_transactions
                if support >= self.min_support:
                    self.frequent_itemsets[candidate] = support
                    self.itemsets_by_level[k].append(candidate)
                    current_frequent.append(candidate)

            k += 1

        return self.frequent_itemsets
