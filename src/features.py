#!/usr/bin/env python3
"""
Feature engineering utilities for the legacy reimbursement model.

This module exposes `make_features` which converts the raw trip inputs
into the engineered feature vector described in the analysis (5-day bump,
piece-wise mileage buckets, receipt cap buckets, etc.).
"""
from typing import List

__all__ = ["make_features", "FEATURE_NAMES"]

FEATURE_NAMES = [
    "days",
    "miles",
    "receipts",
    "is_five",
    "miles_lt400",
    "miles_400_800",
    "miles_gt800",
    "rec_lt1500",
    "rec_gt1500",
    "days_receipts",
    "miles_per_day",
    "rec_gt1500_sq",
]

def make_features(days: float, miles: float, receipts: float) -> List[float]:
    """Return engineered feature vector for a single trip record."""
    # 5-day bonus flag
    is_five = 1.0 if days == 5 else 0.0

    # Piece-wise mileage buckets
    miles_lt400 = min(miles, 400)
    miles_400_800 = max(0.0, min(miles, 800) - 400)
    miles_gt800 = max(0.0, miles - 800)

    # Receipt buckets with cap at 1500
    rec_lt1500 = min(receipts, 1500)
    rec_gt1500 = max(0.0, receipts - 1500)

    # Interaction feature: days * receipts
    days_receipts = days * receipts

    # Miles per day (efficiency)
    miles_per_day = miles / days if days else 0.0

    # Square of high-receipt spill for smoother diminishing returns
    rec_gt1500_sq = rec_gt1500 ** 2

    return [
        days,
        miles,
        receipts,
        is_five,
        miles_lt400,
        miles_400_800,
        miles_gt800,
        rec_lt1500,
        rec_gt1500,
        days_receipts,
        miles_per_day,
        rec_gt1500_sq,
    ] 