"""
Data Preprocessing Module for Market Basket Analysis.

This module loads transaction records from CSV, cleans item strings,
handles missing entries, and groups items into discrete market baskets (sets of items).
"""

import csv
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple


def load_transactions(file_path: str) -> List[Set[str]]:
    """
    Loads transaction records from a CSV file and groups items by transaction ID.

    Supports two common retail CSV schemas:
    1. Long format: TransactionID, [Date], Item (one item per line)
    2. Basket format: TransactionID, Item1;Item2;Item3 or comma-separated items

    Args:
        file_path: Path to the transactions CSV file.

    Returns:
        List of sets, where each set represents the unique items in a transaction basket.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transaction file not found: {file_path}")

    transactions_dict: Dict[str, Set[str]] = defaultdict(set)

    with open(file_path, mode="r", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Transaction file at {file_path} is empty or missing headers.")

        # Identify transaction id and item column names (case-insensitive)
        tid_col = next((c for c in reader.fieldnames if c.strip().lower() in {"transactionid", "tid", "invoice", "invoiceno", "id"}), None)
        item_col = next((c for c in reader.fieldnames if c.strip().lower() in {"item", "itemdescription", "description", "product", "productname", "items"}), None)

        if not tid_col or not item_col:
            # Fallback to positional columns 0 and -1 if headers don't match standard naming
            tid_col = reader.fieldnames[0]
            item_col = reader.fieldnames[-1]

        for row in reader:
            tid = row.get(tid_col, "").strip()
            raw_item = row.get(item_col, "").strip()

            if not tid or not raw_item:
                continue

            # Support delimited items in a single cell (e.g. "Bread, Milk, Butter" or "Bread;Milk")
            if ";" in raw_item:
                items = [it.strip() for it in raw_item.split(";") if it.strip()]
            elif "," in raw_item and item_col.lower() in {"items", "basket"}:
                items = [it.strip() for it in raw_item.split(",") if it.strip()]
            else:
                items = [raw_item]

            for item in items:
                cleaned_item = item.strip().title()
                if cleaned_item:
                    transactions_dict[tid].add(cleaned_item)

    # Convert to list of baskets (each basket is a set of unique items)
    baskets = [items for items in transactions_dict.values() if items]
    return baskets


def compute_dataset_summary(baskets: List[Set[str]]) -> Dict[str, any]:
    """
    Computes summary statistics for the transaction dataset.

    Args:
        baskets: List of transaction item sets.

    Returns:
        Dictionary containing summary statistics.
    """
    total_transactions = len(baskets)
    all_items: Set[str] = set()
    total_items_in_baskets = 0
    item_frequencies: Dict[str, int] = defaultdict(int)

    for basket in baskets:
        all_items.update(basket)
        total_items_in_baskets += len(basket)
        for item in basket:
            item_frequencies[item] += 1

    unique_items_count = len(all_items)
    avg_basket_size = (total_items_in_baskets / total_transactions) if total_transactions > 0 else 0.0
    min_basket_size = min(len(b) for b in baskets) if baskets else 0
    max_basket_size = max(len(b) for b in baskets) if baskets else 0

    return {
        "total_transactions": total_transactions,
        "unique_items_count": unique_items_count,
        "avg_basket_size": avg_basket_size,
        "min_basket_size": min_basket_size,
        "max_basket_size": max_basket_size,
        "item_frequencies": item_frequencies
    }
