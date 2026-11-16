# Market Basket Analysis using Apriori Algorithm

An academic implementation of **Market Basket Analysis (MBA)** and the **Apriori Algorithm** in Python for discovering actionable association rules in retail transaction records.

---

## 1. Project Objective

The objective of this project is to uncover customer purchasing patterns and product affinities from transactional basket data. The implementation:
- Preprocesses raw transaction records into discrete market baskets.
- Implements the Apriori algorithm from first principles, utilizing the downward-closure property for candidate pruning.
- Computes statistical metrics: **Support**, **Confidence**, and **Lift**.
- Generates and ranks association rules (`Antecedent -> Consequent`) based on predictive strength and correlation.
- Provides a configurable command-line interface for tuning thresholds.

---

## 2. Mathematical Background & Evaluation Metrics

Given a collection of transactions $\mathcal{D}$ and an itemset $I = A \cup B$:

### 1. Support
The proportion of transactions in the dataset that contain the itemset:
$$\text{Support}(A \to B) = P(A \cup B) = \frac{\text{Count}(A \cup B)}{|\mathcal{D}|}$$

### 2. Confidence
The conditional probability that a customer purchases itemset $B$ given that they have already purchased itemset $A$:
$$\text{Confidence}(A \to B) = P(B \mid A) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$$

### 3. Lift
The ratio of observed support to the expected support if $A$ and $B$ were statistically independent:
$$\text{Lift}(A \to B) = \frac{\text{Confidence}(A \to B)}{\text{Support}(B)} = \frac{P(A \cup B)}{P(A) \cdot P(B)}$$
- **$\text{Lift} = 1$:** $A$ and $B$ are independent; no actionable relationship.
- **$\text{Lift} > 1$:** Positive correlation; purchasing $A$ noticeably increases the likelihood of buying $B$.
- **$\text{Lift} < 1$:** Negative correlation (substitute goods).

---

## 3. Dataset Notice

The included dataset at `data/grocery_transactions.csv` is a clearly labeled **illustrative benchmark grocery dataset** modeled after standard point-of-sale retail records. It simulates realistic basket co-occurrences (e.g., breakfast staples, dairy, beverages, and snack pairings) to validate the Apriori mining pipeline.

---

## 4. File Structure

```
Market-Basket-Analysis-on-real-world-data/
├── README.md                     # Documentation and algorithm explanation
├── requirements.txt              # Environment dependencies
├── main.py                       # Application entry point and CLI runner
├── data_preprocessing.py         # CSV ingestion, data cleaning, and basket aggregation
├── apriori.py                    # Core Apriori algorithm (L1, candidate Ck, pruning, Lk)
├── association_rules.py          # Association rule generation (Support, Confidence, Lift)
├── data/
│   └── grocery_transactions.csv  # Illustrative grocery transaction benchmark dataset
└── .gitignore                    # Standard Python gitignore rules
```

---

## 5. How the Program Works

```
[ Transactions CSV ]
         │
         ▼
┌─────────────────────────┐
│  DATA PREPROCESSING     │  Loads records, cleans strings, groups
└───────────┬─────────────┘  by TransactionID into itemset baskets.
         │
         ▼
┌─────────────────────────┐
│     APRIORI MINER       │  Iteratively mines frequent itemsets (L1 -> Lk)
└───────────┬─────────────┘  using downward-closure subset pruning.
         │
         ▼
┌─────────────────────────┐
│  ASSOCIATION RULES      │  Generates candidate subsets, computes
└───────────┬─────────────┘  Support, Confidence, Lift, filters by thresholds.
         │
         ▼
[ Formatted Table & Retail Recommendations ]
```

---

## 6. Execution Instructions

### Running the default analysis:
```bash
python main.py
```

### Configuring thresholds via CLI:
```bash
# Set custom minimum support (8%) and minimum confidence (60%)
python main.py --min-support 0.08 --min-confidence 0.60

# Filter rules with minimum lift of 1.5 and display top 10 rules
python main.py --min-support 0.10 --min-confidence 0.50 --min-lift 1.5 --top-n 10

# Run on a custom transaction CSV dataset
python main.py --input /path/to/custom_transactions.csv
```

---

## 7. Expected Output

```
==============================================================================
 DATASET SUMMARY & TRANSACTION STATISTICS
==============================================================================
Total Transactions Recorded : 40
Unique Products Identified  : 16
Average Items per Basket    : 2.45
Basket Size Range           : 2 to 4 items

Top 5 Most Purchased Items:
  1. Bread                :  18 transactions ( 45.0% support)
  2. Whole Milk           :  16 transactions ( 40.0% support)
  3. Butter               :  15 transactions ( 37.5% support)
  4. Beer                 :  10 transactions ( 25.0% support)
  5. Potato Chips         :   7 transactions ( 17.5% support)

==============================================================================
 FREQUENT ITEMSETS DISCOVERED (MIN SUPPORT = 10.0%)
==============================================================================
Total Frequent Itemsets Found: 22

--- Level 1 Itemsets (Size 1) : Found 12 ---
  (Bread) -> Support: 0.450 (45.0%)
  (Whole Milk) -> Support: 0.400 (40.0%)
  (Butter) -> Support: 0.375 (37.5%)
  ...

--- Level 2 Itemsets (Size 2) : Found 9 ---
  (Bread, Butter) -> Support: 0.350 (35.0%)
  (Bread, Whole Milk) -> Support: 0.225 (22.5%)
  (Beer, Potato Chips) -> Support: 0.175 (17.5%)
  ...

==============================================================================
 DISCOVERED ASSOCIATION RULES (SORTED BY LIFT)
==============================================================================
No.  | Antecedent (IF)           | Consequent (THEN)    | Support  | Conf     | Lift    
------------------------------------------------------------------------------
1    | Coffee                    | Sugar                |  0.125   |  1.000   |  8.000
2    | Sugar                     | Coffee               |  0.125   |  1.000   |  8.000
3    | Pasta                     | Tomato Sauce         |  0.125   |  1.000   |  8.000
4    | Tomato Sauce              | Pasta                |  0.125   |  1.000   |  8.000
5    | Potato Chips              | Beer                 |  0.125   |  1.000   |  5.000
6    | Beer                      | Potato Chips         |  0.125   |  0.625   |  5.000
7    | Eggs                      | Bread, Whole Milk    |  0.100   |  0.667   |  3.333
8    | Butter                    | Bread                |  0.300   |  1.000   |  2.500
9    | Butter, Whole Milk        | Bread                |  0.125   |  1.000   |  2.500
10   | Bread                     | Butter               |  0.300   |  0.750   |  2.500
...
```
