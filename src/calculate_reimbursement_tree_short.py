#!/usr/bin/env python3
"""Auto-generated decision tree scorer."""
from typing import Union

def calculate_reimbursement_tree_short(trip_duration_days: Union[int,float],
                miles_traveled: Union[int,float],
                total_receipts_amount: Union[int,float]) -> float:
    days = float(trip_duration_days)
    miles = float(miles_traveled)
    receipts = float(total_receipts_amount)
    # replicate feature engineering inline for speed
    days_receipts = days * receipts
    miles_lt400 = min(miles, 400.0)
    miles_400_800 = max(0.0, min(miles, 800.0) - 400.0)
    miles_gt800 = max(0.0, miles - 800.0)
    rec_lt1500 = min(receipts, 1500.0)
    rec_gt1500 = max(0.0, receipts - 1500.0)
    miles_per_day = miles / days if days else 0.0
    rec_gt1500_sq = rec_gt1500 ** 2
    # positional binding (names generated for clarity only)
    days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq = (
        days, miles, receipts, 1.0 if days==5 else 0.0,
        miles_lt400, miles_400_800, miles_gt800,
        rec_lt1500, rec_gt1500, days_receipts,
        miles_per_day, rec_gt1500_sq
    )
    if rec_lt1500 <= 857.644989:
        if miles_400_800 <= 183.000000:
            if rec_lt1500 <= 562.035004:
                if days <= 1.500000:
                    if miles_per_day <= 197.500000:
                        return 161.461818
                    else:
                        return 297.590833
                else:
                    if miles <= 203.500000:
                        if days_receipts <= 63.330000:
                            return 318.676000
                        else:
                            return 415.240909
                    else:
                        return 545.694118
            else:
                return 704.786875
        else:
            if receipts <= 563.100006:
                if days <= 2.500000:
                    return 625.386111
                else:
                    if miles_per_day <= 314.550003:
                        return 784.407500
                    else:
                        return 757.905000
            else:
                return 1022.828571
    else:
        if receipts <= 1074.315002:
            if miles_lt400 <= 351.500000:
                return 963.042727
            else:
                return 1183.140000
        else:
            if miles <= 554.000000:
                if days_receipts <= 5098.540039:
                    if days <= 1.500000:
                        return 1178.565000
                    else:
                        return 1298.580000
                else:
                    if miles_lt400 <= 215.000000:
                        return 1299.958333
                    else:
                        return 1475.552632
            else:
                if days_receipts <= 2271.224976:
                    if miles_gt800 <= 191.000000:
                        return 1385.778824
                    else:
                        return 1307.417000
                else:
                    if days <= 3.500000:
                        if miles <= 739.000000:
                            return 1453.244545
                        else:
                            if rec_gt1500 <= 582.755005:
                                return 1518.575000
                            else:
                                return 1488.847692
                    else:
                        return 1619.912667
