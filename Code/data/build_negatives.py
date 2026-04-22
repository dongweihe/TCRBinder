#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-faithful negative sampling for TCRBinder.

Implements the internal-mismatching negative sampling described in the paper
(Methods, "Datasets" section):

    "To construct the negative dataset, we adopted an internal mismatching
     strategy with a 1:5 positive-to-negative ratio. Specifically, for each
     positive triple, we fixed the peptide (and its associated HLA allele)
     and paired it with five TCRs randomly sampled from the training dataset
     that were originally known to bind different peptides. To minimize the
     risk of potential cross-reactivity (false negatives), we ensured that
     the sampled TCRs were derived from peptides with low sequence similarity
     (Levenshtein distance > 3) to the target peptide."

Guarantees enforced here, in addition to the paper's requirements:

    * Each split (train / val / test) produces its own negatives from its own
      positive pool. The TCR donor pool for a given split is drawn only from
      that same split, so negatives never leak TCR identities across splits.
    * No generated (TCR_Beta, TCR_Alpha, Antigen, HLA) negative tuple is ever
      observed as a positive anywhere in the union of the three splits.
    * Exact duplicate negative tuples within a split are deduplicated.

Usage
-----
As a library::

    from data.build_negatives import build_negatives
    train_full, val_full, test_full = build_negatives(
        train_pos_df, val_pos_df, test_pos_df,
        ratio=5, peptide_lev_threshold=3, seed=7,
    )

As a CLI (reads a positive CSV, splits 3:1:1, writes PHT.csv)::

    python -m data.build_negatives \
        --input ../ProcessedData/PHT_positive.csv \
        --output ../PHT.csv \
        --ratio 5 --peptide-lev-threshold 3 --seed 7
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("tcrbinder.negatives")

# Column names consistent with Sample.csv / PHT.csv
COL_BETA = "TCR_Beta"
COL_ALPHA = "TCR_Alpha"
COL_ANTIGEN = "Antigen"
COL_HLA = "HLA"
COL_LABEL = "Label"
TRIPLE_COLS = (COL_BETA, COL_ALPHA, COL_ANTIGEN, COL_HLA)


# ---------------------------------------------------------------------------
# Levenshtein distance (pure-python, no extra deps)
# ---------------------------------------------------------------------------
def _levenshtein(a: str, b: str) -> int:
    """Classic iterative Levenshtein distance."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            insert_ = current[j - 1] + 1
            delete_ = previous[j] + 1
            sub_ = previous[j - 1] + (ca != cb)
            current[j] = min(insert_, delete_, sub_)
        previous = current
    return previous[-1]


def _compatible_peptide_mask(
    target_peptide: str,
    candidate_peptides: np.ndarray,
    threshold: int,
) -> np.ndarray:
    """Return a boolean mask of peptides that are at Levenshtein > threshold
    from ``target_peptide``.
    """
    distances = np.fromiter(
        (_levenshtein(target_peptide, p) for p in candidate_peptides),
        dtype=np.int32,
        count=len(candidate_peptides),
    )
    return distances > threshold


# ---------------------------------------------------------------------------
# Negative sampling for a single split
# ---------------------------------------------------------------------------
def _sample_negatives_for_split(
    pos_df: pd.DataFrame,
    *,
    ratio: int,
    peptide_lev_threshold: int,
    global_positive_set: set,
    rng: np.random.Generator,
    split_name: str,
) -> pd.DataFrame:
    """Generate negatives for one split using ONLY its own positive pool.

    The donor TCR pool (beta/alpha pair) is drawn from the same split's
    positives, so TCR identities never leak across splits.
    """
    if pos_df.empty:
        LOGGER.warning("[%s] Positive set is empty; skipping negative generation.", split_name)
        return pos_df.copy()

    # Donor pool: unique (TCR_Beta, TCR_Alpha) paired with the peptide they
    # are known to bind (needed to check the Levenshtein > threshold rule
    # "sampled TCRs were derived from peptides with low sequence similarity").
    donor_df = (
        pos_df[[COL_BETA, COL_ALPHA, COL_ANTIGEN]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    donor_peptides = donor_df[COL_ANTIGEN].to_numpy()
    donor_indices = np.arange(len(donor_df))

    out_records: List[dict] = []
    seen_in_split: set = set()

    for _, row in pos_df.iterrows():
        target_peptide = row[COL_ANTIGEN]
        target_hla = row[COL_HLA]

        mask = _compatible_peptide_mask(target_peptide, donor_peptides, peptide_lev_threshold)
        eligible = donor_indices[mask]
        if eligible.size == 0:
            LOGGER.debug(
                "[%s] No donor TCR satisfies Levenshtein > %d for peptide %s; "
                "skipping this positive.",
                split_name, peptide_lev_threshold, target_peptide,
            )
            continue

        attempts, collected = 0, 0
        max_attempts = ratio * 20  # bounded resampling to avoid infinite loops
        while collected < ratio and attempts < max_attempts:
            pick = int(rng.choice(eligible))
            attempts += 1
            donor = donor_df.iloc[pick]
            beta, alpha = donor[COL_BETA], donor[COL_ALPHA]

            triple = (beta, alpha, target_peptide, target_hla)
            if triple in global_positive_set:
                continue  # would be a false negative
            if triple in seen_in_split:
                continue  # in-split duplicate
            seen_in_split.add(triple)

            out_records.append({
                COL_BETA: beta,
                COL_ALPHA: alpha,
                COL_ANTIGEN: target_peptide,
                COL_HLA: target_hla,
                COL_LABEL: 0,
            })
            collected += 1

        if collected < ratio:
            LOGGER.debug(
                "[%s] Only produced %d/%d negatives for peptide %s (eligible donors=%d).",
                split_name, collected, ratio, target_peptide, int(eligible.size),
            )

    neg_df = pd.DataFrame(out_records, columns=[COL_BETA, COL_ALPHA, COL_ANTIGEN, COL_HLA, COL_LABEL])
    LOGGER.info(
        "[%s] Generated %d negatives for %d positives (observed ratio=1:%.2f).",
        split_name, len(neg_df), len(pos_df), (len(neg_df) / max(len(pos_df), 1)),
    )
    return neg_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_negatives(
    train_pos_df: pd.DataFrame,
    val_pos_df: pd.DataFrame,
    test_pos_df: pd.DataFrame,
    *,
    ratio: int = 5,
    peptide_lev_threshold: int = 3,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the negative dataset for the three positive splits.

    Returns three dataframes (train/val/test) that each contain the original
    positives followed by the newly sampled negatives.
    """
    required = {COL_BETA, COL_ALPHA, COL_ANTIGEN, COL_HLA}
    for name, df in (("train", train_pos_df), ("val", val_pos_df), ("test", test_pos_df)):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"[{name}] positive dataframe is missing columns: {sorted(missing)}")

    # Ensure label column exists and is set to 1 for positives
    for df in (train_pos_df, val_pos_df, test_pos_df):
        df[COL_LABEL] = 1

    # Build the union-of-positives set once so negatives cannot overlap any
    # known positive (anywhere in train/val/test).
    all_pos = pd.concat([train_pos_df, val_pos_df, test_pos_df], ignore_index=True)
    global_positive_set: set = set(
        zip(
            all_pos[COL_BETA].tolist(),
            all_pos[COL_ALPHA].tolist(),
            all_pos[COL_ANTIGEN].tolist(),
            all_pos[COL_HLA].tolist(),
        )
    )

    rng = np.random.default_rng(seed)

    outs = []
    for name, df in (
        ("train", train_pos_df),
        ("val", val_pos_df),
        ("test", test_pos_df),
    ):
        neg_df = _sample_negatives_for_split(
            df,
            ratio=ratio,
            peptide_lev_threshold=peptide_lev_threshold,
            global_positive_set=global_positive_set,
            rng=rng,
            split_name=name,
        )
        full = pd.concat([df, neg_df], ignore_index=True)
        outs.append(full)

    _sanity_checks(outs[0], outs[1], outs[2])
    return outs[0], outs[1], outs[2]


# ---------------------------------------------------------------------------
# Sanity checks — intentionally strict; they are cheap and prevent regressions
# ---------------------------------------------------------------------------
def _sanity_checks(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    pos = all_df[all_df[COL_LABEL] == 1]
    neg = all_df[all_df[COL_LABEL] == 0]

    pos_triples = set(map(tuple, pos[list(TRIPLE_COLS)].to_numpy()))
    neg_triples_list = list(map(tuple, neg[list(TRIPLE_COLS)].to_numpy()))
    neg_triples = set(neg_triples_list)

    overlap = pos_triples & neg_triples
    assert not overlap, f"Negatives overlap positives on {len(overlap)} tuples"

    assert len(neg_triples_list) == len(neg_triples), (
        f"Duplicate negatives detected: {len(neg_triples_list) - len(neg_triples)} duplicates"
    )

    LOGGER.info("Sanity checks passed: no pos/neg overlap, no duplicated negatives.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _random_split_positive(
    pos_df: pd.DataFrame, ratios: Tuple[float, float, float], seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(sum(ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    rng = np.random.default_rng(seed)
    idx = np.arange(len(pos_df))
    rng.shuffle(idx)
    n_train = int(len(idx) * ratios[0])
    n_val = int(len(idx) * ratios[1])
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return (
        pos_df.iloc[train_idx].reset_index(drop=True),
        pos_df.iloc[val_idx].reset_index(drop=True),
        pos_df.iloc[test_idx].reset_index(drop=True),
    )


def _main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="CSV of positive triples (Label must be 1 or absent).")
    parser.add_argument("--output", required=True, help="Output CSV (positives + negatives, all splits merged).")
    parser.add_argument("--ratio", type=int, default=5, help="1:N positive-to-negative ratio (default 5).")
    parser.add_argument(
        "--peptide-lev-threshold", type=int, default=3,
        help="Sampled TCRs must come from peptides with Levenshtein distance > this value.",
    )
    parser.add_argument(
        "--train-val-test", type=float, nargs=3, default=(0.6, 0.2, 0.2),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Paper default is 3:1:1 (0.6 0.2 0.2).",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-column", default=None, help="Optional column name to write split labels to.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    pos_df = pd.read_csv(args.input)
    if COL_LABEL in pos_df.columns:
        pos_df = pos_df[pos_df[COL_LABEL] == 1].reset_index(drop=True)
        pos_df = pos_df.drop(columns=[COL_LABEL])

    LOGGER.info(
        "Loaded %d positive triples; splitting %s with seed=%d",
        len(pos_df), args.train_val_test, args.seed,
    )

    train_pos, val_pos, test_pos = _random_split_positive(
        pos_df, tuple(args.train_val_test), seed=args.seed
    )
    train_full, val_full, test_full = build_negatives(
        train_pos, val_pos, test_pos,
        ratio=args.ratio,
        peptide_lev_threshold=args.peptide_lev_threshold,
        seed=args.seed,
    )

    if args.split_column:
        train_full[args.split_column] = "train"
        val_full[args.split_column] = "val"
        test_full[args.split_column] = "test"

    merged = pd.concat([train_full, val_full, test_full], ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    merged.to_csv(args.output, index=False)
    LOGGER.info(
        "Wrote %d rows to %s (train=%d, val=%d, test=%d).",
        len(merged), args.output, len(train_full), len(val_full), len(test_full),
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
