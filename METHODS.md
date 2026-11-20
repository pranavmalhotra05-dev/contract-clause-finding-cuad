# Methods and decisions

**Data.** CUAD v1 JSON (SQuAD format): 510 contracts, 41 question types, answers as character spans. Only the JSON is used.

**Segments.** Split on sentence-like boundaries (`.`, `;`, `:` followed by an upper-case letter, digit, bracket or quote), merge pieces under 40 characters, cut pieces over 900 characters. 107,102 segments.

**Labels.** Segment s is positive for category c if for some expert span the overlap is at least half of the shorter of the two. A first rule (overlap at least half of the segment) missed 26% of spans; this one touches 13,301 of 13,302 spans in the 39 categories studied.

**Categories studied.** 39 of 41: Document Name and Parties are excluded because they are not clause-finding tasks.

**Scorers.**
- Keyword rules: one regular expression per category for 18 categories (see `src/04_models.py`), ties broken at random.
- TF-IDF (word 1-2 grams, min document frequency 3, 60,000 features, English stop words) with logistic regression (C=4, class-balanced).
- all-MiniLM-L6-v2 sentence embeddings (frozen, 384 dimensions, max 256 tokens) with the same logistic regression.
- Training uses all positives and a random 12 times as many negatives per fold; scoring uses every held-out segment.

**Splits (5 folds).** Random segments; random contracts; GroupKFold by company (first token of the contract name); GroupKFold by normalized agreement type (the name after the last hyphen, lower-cased, numeric prefixes removed).

**Metrics.** Segment AUPR over pooled out-of-fold scores; contract AUPR of the maximum segment score against "contract has the clause"; hit at 3 over contracts that have the clause. Macro-average over categories.

**Known issues.**
- Single seed; no confidence intervals. Differences of about 0.01 should be read as noise.
- Company and agreement type are derived from file names and are only approximate groupings of related contracts.
- Segments can cut a clause that spans several sentences; the label rule is a compromise.
- Rules were written once, before seeing results for these categories, with no tuning.
