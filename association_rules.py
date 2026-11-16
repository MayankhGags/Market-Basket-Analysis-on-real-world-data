"""
Association Rules Mining Module.

Generates association rules (Antecedent -> Consequent) from frequent itemsets.
Calculates Support, Confidence, and Lift for each candidate rule.

Key Metrics:
1. Support(A -> B) = P(A ∩ B) = Count(A ∩ B) / Total_Transactions
2. Confidence(A -> B) = P(B | A) = Support(A ∩ B) / Support(A)
3. Lift(A -> B) = Confidence(A -> B) / Support(B) = P(A ∩ B) / (P(A) * P(B))
   - Lift = 1: A and B are independent.
   - Lift > 1: Positive association (buying A increases likelihood of buying B).
   - Lift < 1: Negative association (substitute goods).
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, FrozenSet, List, Set


@dataclass
class AssociationRule:
    """Represents an association rule with key evaluation metrics."""
    antecedent: FrozenSet[str]
    consequent: FrozenSet[str]
    support: float
    confidence: float
    lift: float

    def antecedent_str(self) -> str:
        return ", ".join(sorted(self.antecedent))

    def consequent_str(self) -> str:
        return ", ".join(sorted(self.consequent))

    def __repr__(self) -> str:
        return (
            f"{{{self.antecedent_str()}}} -> {{{self.consequent_str()}}} "
            f"[Support={self.support:.3f}, Conf={self.confidence:.3f}, Lift={self.lift:.3f}]"
        )


class RuleGenerator:
    """Generates and evaluates association rules from frequent itemsets."""

    def __init__(self, min_confidence: float = 0.5, min_lift: float = 1.0):
        """
        Args:
            min_confidence: Minimum confidence threshold (0.0 < min_confidence <= 1.0).
            min_lift: Minimum lift threshold (typically >= 1.0 for positive association).
        """
        if not (0.0 < min_confidence <= 1.0):
            raise ValueError("min_confidence must be in range (0.0, 1.0].")
        self.min_confidence = min_confidence
        self.min_lift = min_lift

    def generate_rules(
        self, frequent_itemsets: Dict[FrozenSet[str], float]
    ) -> List[AssociationRule]:
        """
        Generates all valid association rules from the provided frequent itemsets.

        Args:
            frequent_itemsets: Dictionary of {frozenset(items): support}.

        Returns:
            List of generated AssociationRule instances matching thresholds.
        """
        rules: List[AssociationRule] = []

        # Association rules can only be formed from itemsets with at least 2 items
        multi_itemsets = [itemset for itemset in frequent_itemsets if len(itemset) >= 2]

        for itemset in multi_itemsets:
            itemset_support = frequent_itemsets[itemset]
            item_list = list(itemset)
            itemset_len = len(itemset)

            # Generate all non-empty proper subsets as antecedents
            for r in range(1, itemset_len):
                for antecedent_tuple in combinations(item_list, r):
                    antecedent = frozenset(antecedent_tuple)
                    consequent = itemset - antecedent

                    if antecedent in frequent_itemsets and consequent in frequent_itemsets:
                        antecedent_support = frequent_itemsets[antecedent]
                        consequent_support = frequent_itemsets[consequent]

                        # Calculate Confidence
                        confidence = itemset_support / antecedent_support

                        # Filter by min_confidence
                        if confidence >= self.min_confidence:
                            # Calculate Lift
                            lift = confidence / consequent_support if consequent_support > 0 else 0.0

                            if lift >= self.min_lift:
                                rule = AssociationRule(
                                    antecedent=antecedent,
                                    consequent=consequent,
                                    support=itemset_support,
                                    confidence=confidence,
                                    lift=lift
                                )
                                rules.append(rule)

        # Sort rules descending by Lift, then Confidence, then Support
        rules.sort(key=lambda r: (r.lift, r.confidence, r.support), reverse=True)
        return rules
