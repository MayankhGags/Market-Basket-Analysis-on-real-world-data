"""
Market Basket Analysis using Apriori Algorithm.

This application performs association rule mining on retail transaction data:
1. Ingests and cleans retail transaction baskets.
2. Identifies frequent itemsets via the Apriori algorithm.
3. Computes Support, Confidence, and Lift metrics.
4. Generates and displays actionable association rules.
"""

import argparse
import os
import sys
from typing import Dict, FrozenSet, List

from apriori import AprioriMiner
from association_rules import AssociationRule, RuleGenerator
from data_preprocessing import compute_dataset_summary, load_transactions


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(f" {title.upper()}")
    print("=" * 78)


def display_dataset_summary(summary: Dict[str, any]) -> None:
    print_header("Dataset Summary & Transaction Statistics")
    print(f"Total Transactions Recorded : {summary['total_transactions']}")
    print(f"Unique Products Identified  : {summary['unique_items_count']}")
    print(f"Average Items per Basket    : {summary['avg_basket_size']:.2f}")
    print(f"Basket Size Range           : {summary['min_basket_size']} to {summary['max_basket_size']} items")

    print("\nTop 5 Most Purchased Items:")
    sorted_items = sorted(summary["item_frequencies"].items(), key=lambda x: x[1], reverse=True)[:5]
    for rank, (item, count) in enumerate(sorted_items, 1):
        freq_pct = (count / summary["total_transactions"]) * 100
        print(f"  {rank}. {item:<20} : {count:3d} transactions ({freq_pct:5.1f}% support)")


def display_frequent_itemsets(miner: AprioriMiner) -> None:
    print_header(f"Frequent Itemsets Discovered (Min Support = {miner.min_support:.1%})")
    total_frequent = len(miner.frequent_itemsets)
    print(f"Total Frequent Itemsets Found: {total_frequent}\n")

    for level, itemsets in sorted(miner.itemsets_by_level.items()):
        print(f"--- Level {level} Itemsets (Size {level}) : Found {len(itemsets)} ---")
        sorted_level = sorted(itemsets, key=lambda s: miner.frequent_itemsets[s], reverse=True)
        for itemset in sorted_level[:10]:
            items_str = ", ".join(sorted(itemset))
            sup = miner.frequent_itemsets[itemset]
            print(f"  ({items_str}) -> Support: {sup:.3f} ({sup:.1%})")
        if len(sorted_level) > 10:
            print(f"  ... and {len(sorted_level) - 10} more itemsets.")
        print()


def display_association_rules(rules: List[AssociationRule], top_n: int = 15) -> None:
    print_header("Discovered Association Rules (Sorted by Lift)")
    if not rules:
        print("No association rules found matching the specified confidence and support thresholds.")
        print("Try lowering --min-support or --min-confidence.")
        return

    print(f"Found {len(rules)} valid rules. Displaying top {min(len(rules), top_n)} rules:\n")
    print(f"{'No.':<4} | {'Antecedent (IF)':<25} | {'Consequent (THEN)':<20} | {'Support':<8} | {'Conf':<8} | {'Lift':<8}")
    print("-" * 78)

    for idx, rule in enumerate(rules[:top_n], 1):
        ant = rule.antecedent_str()
        con = rule.consequent_str()
        print(f"{idx:<4} | {ant:<25} | {con:<20} | {rule.support:6.3f}   | {rule.confidence:6.3f}   | {rule.lift:6.3f}")

    print("-" * 78)
    print("\nActionable Retail Insights (Top Rules):")
    for idx, rule in enumerate(rules[:3], 1):
        print(f"  - Insight {idx}: Customers purchasing [{rule.antecedent_str()}] are {rule.confidence:.1%}")
        print(f"    likely to also purchase [{rule.consequent_str()}] ({rule.lift:.2f}x higher than baseline rate).")


def run_market_basket_analysis(
    file_path: str,
    min_support: float,
    min_confidence: float,
    min_lift: float,
    top_n: int
) -> None:
    """Executes the complete Market Basket Analysis workflow."""
    print(f"[Market Basket Analysis] Loading transactions from: {file_path}")
    baskets = load_transactions(file_path)
    summary = compute_dataset_summary(baskets)
    display_dataset_summary(summary)

    # 1. Run Apriori Miner
    miner = AprioriMiner(min_support=min_support)
    miner.fit(baskets)
    display_frequent_itemsets(miner)

    # 2. Generate Association Rules
    generator = RuleGenerator(min_confidence=min_confidence, min_lift=min_lift)
    rules = generator.generate_rules(miner.frequent_itemsets)
    display_association_rules(rules, top_n=top_n)


def main() -> None:
    default_dataset = os.path.join(os.path.dirname(__file__), "data", "grocery_transactions.csv")
    parser = argparse.ArgumentParser(description="Market Basket Analysis using Apriori Algorithm")
    parser.add_argument("--input", "-i", default=default_dataset, help=f"Path to transactions CSV (default: {default_dataset})")
    parser.add_argument("--min-support", "-s", type=float, default=0.10, help="Minimum support threshold [0.0 - 1.0] (default: 0.10)")
    parser.add_argument("--min-confidence", "-c", type=float, default=0.50, help="Minimum confidence threshold [0.0 - 1.0] (default: 0.50)")
    parser.add_argument("--min-lift", "-l", type=float, default=1.0, help="Minimum lift threshold (default: 1.0)")
    parser.add_argument("--top-n", "-n", type=int, default=15, help="Number of rules to display (default: 15)")

    args = parser.parse_args()

    try:
        run_market_basket_analysis(
            file_path=args.input,
            min_support=args.min_support,
            min_confidence=args.min_confidence,
            min_lift=args.min_lift,
            top_n=args.top_n
        )
    except Exception as e:
        print(f"Error during Market Basket Analysis: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
