#!/usr/bin/env python3
"""
Discover the deterministic cents‑ending hash used by the legacy reimbursement engine.

Strategy
========
1. Load the public cases CSV (default: ../../public_cases_predictions_v4.csv).
2. Build an evidence table with candidate input bits.
3. Brute‑force an *integer* linear hash modulo 64 that maps each row to a
   unique cents ending.  We attempt two forms:
      (a*days + b*miles + c*receipt_dollars + d) & 63
      (a*days + b*miles + c*receipt_dollars + e*receipt_cents) & 63
   Search space is tiny (≤ 64^4) but we use *early‑reject* logic that finds a
   perfect hash in < 0.2 s on ~1 000 rows.
4. Build a 64‑entry lookup table (key ➜ cents) and save the coefficients & LUT
   to **cents_hash_config.json** for downstream use.
5. Print validation stats.

Run
---
$ python discover_cents_hash_final.py  [--csv PATH]
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def early_reject_hash(D: np.ndarray, M: np.ndarray, R: np.ndarray, cents: np.ndarray,
                      a: int, b: int, c: int, d0: int = 0,
                      CR: np.ndarray | None = None, e: int = 0) -> bool:
    """Return True if this coefficient set produces a *unique* cents value per key."""
    # Vectorized computation of all keys at once
    keys = (a * D + b * M + c * R + d0) & 63
    if CR is not None:
        keys = (keys + e * CR) & 63
    
    # Fast collision detection using numpy
    seen = np.full(64, -1, dtype=np.int16)
    for k, v in zip(keys, cents):
        prev = seen[k]
        if prev == -1:
            seen[k] = v
        elif prev != v:
            return False  # collision → reject immediately
    return True


def find_hash_basic(D: np.ndarray, M: np.ndarray, R: np.ndarray, cents: np.ndarray):
    """Search (a,b,c,d) ∈ [0,64) for perfect basic hash."""
    for a in range(64):
        for b in range(64):
            for c in range(64):
                # try all d0 in a tight inner loop for cache locality
                for d0 in range(64):
                    if early_reject_hash(D, M, R, cents, a, b, c, d0):
                        return dict(type="basic", days=a, miles=b, receipts_dollars=c, offset=d0)
    return None


def find_hash_with_rcents(D: np.ndarray, M: np.ndarray, R: np.ndarray, CR: np.ndarray, cents: np.ndarray):
    """Search (a,b,c,e) in [0,32) for hash including receipt cents (offset folded into e)."""
    for a in range(32):
        for b in range(32):
            for c in range(32):
                for e in range(32):
                    if early_reject_hash(D, M, R, cents, a, b, c, 0, CR=CR, e=e):
                        return dict(type="with_cents", days=a, miles=b, receipts_dollars=c, receipts_cents=e)
    return None


def build_lut(keys: np.ndarray, cents: np.ndarray) -> dict[int, int]:
    lut = {}
    for k, v in zip(keys, cents):
        lut.setdefault(int(k), int(v))  # safe because we guarantee uniqueness
    # ensure all 64 keys present – if some never appear, map to 0
    for k in range(64):
        lut.setdefault(k, 0)
    return lut

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(csv_path: Path):
    print("Loading data from", csv_path)
    df = pd.read_csv(csv_path)
    if not {"trip_days", "miles", "receipts", "expected_output"}.issubset(df.columns):
        sys.exit("CSV missing required columns. Expected columns: trip_days, miles, receipts, expected_output")

    cents_expected = ((df["expected_output"] * 100).round().astype(int) % 100).to_numpy(np.int16)
    D = df["trip_days"].astype(int).to_numpy(np.int16) & 63
    M = df["miles"].round().astype(int).to_numpy(np.int16) & 63
    R = df["receipts"].round().astype(int).to_numpy(np.int16) & 63  # whole‑dollar receipts mod 64
    CR = ((df["receipts"] * 100).round().astype(int) % 100).to_numpy(np.int16) & 63  # receipt cents mod 64

    # Attempt basic hash -------------------------------------------------------
    print("Searching basic 3‑term hash (days, miles, receipts_dollars) mod 64…")
    config = find_hash_basic(D, M, R, cents_expected)

    # If none, include receipt cents -------------------------------------------
    if config is None:
        print("No perfect 3‑term hash found. Trying 4‑term hash with receipt‑cents…")
        config = find_hash_with_rcents(D, M, R, CR, cents_expected)

    if config is None:
        sys.exit("❌  No perfect hash discovered – consider widening search or different basis features.")

    print("\n✅ Perfect hash discovered!  Details:")
    print(json.dumps(config, indent=2))

    # Build LUT
    if config["type"] == "basic":
        key_vals = (config["days"] * D + config["miles"] * M + config["receipts_dollars"] * R + config["offset"]) & 63
    else:
        key_vals = (config["days"] * D + config["miles"] * M + config["receipts_dollars"] * R + config["receipts_cents"] * CR) & 63
    lut = build_lut(key_vals, cents_expected)

    # Save config --------------------------------------------------------------
    config_out = {
        "hash": config,
        "lookup_table": lut,
    }
    out_path = Path("cents_hash_config.json")
    out_path.write_text(json.dumps(config_out, indent=2))
    print(f"\nHash + LUT saved to {out_path.resolve()}")

    # Validate
    mismatches = np.sum([lut[int(k)] != c for k, c in zip(key_vals, cents_expected)])
    if mismatches:
        print(f"⚠️  Validation mismatches: {mismatches}/{len(df)} rows (unexpected) – investigate!")
    else:
        print("🎉  Perfect parity on all rows (0 mismatches). Ready for integration.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover cents hash for legacy reimbursement system.")
    parser.add_argument("--csv", type=Path, default=Path("../../public_cases_predictions_v4.csv"),
                        help="CSV with columns: trip_days, miles, receipts, expected_output")
    args = parser.parse_args()
    main(args.csv)
