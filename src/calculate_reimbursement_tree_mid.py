#!/usr/bin/env python3
"""Auto-generated decision tree scorer."""
from typing import Union

def calculate_reimbursement_tree_mid(trip_duration_days: Union[int,float],
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
    if days_receipts <= 4343.074951:
        if miles_lt400 <= 368.500000:
            if miles <= 263.934998:
                if days_receipts <= 2539.604980:
                    return 580.856111
                else:
                    return 750.218182
            else:
                return 796.413000
        else:
            if receipts <= 500.494995:
                if miles <= 628.500000:
                    return 867.505333
                else:
                    if days <= 5.500000:
                        return 951.156429
                    else:
                        return 1089.407500
            else:
                if miles_per_day <= 142.650002:
                    return 1145.217000
                else:
                    return 1256.148000
    else:
        if miles_per_day <= 103.512501:
            if days_receipts <= 6179.500000:
                if miles_per_day <= 51.387501:
                    return 1000.779231
                else:
                    return 1318.368182
            else:
                if days <= 7.500000:
                    if is_five <= 0.500000:
                        if miles_lt400 <= 370.000000:
                            if rec_gt1500 <= 171.789993:
                                return 1509.147000
                            else:
                                return 1581.839000
                        else:
                            return 1680.269333
                    else:
                        return 1470.431333
                else:
                    if miles_per_day <= 64.125000:
                        if receipts <= 1460.190002:
                            return 1376.105833
                        else:
                            return 1491.959091
                    else:
                        return 1509.921667
        else:
            if days_receipts <= 6072.410156:
                if days_receipts <= 4691.665039:
                    return 1467.755000
                else:
                    return 1553.962308
            else:
                if days <= 6.500000:
                    if is_five <= 0.500000:
                        return 1792.690556
                    else:
                        if miles_400_800 <= 359.000000:
                            return 1688.533000
                        else:
                            if receipts <= 2073.540039:
                                return 1737.196000
                            else:
                                return 1706.451000
                else:
                    if days_receipts <= 12901.875000:
                        if miles_per_day <= 127.562500:
                            return 1904.097000
                        else:
                            return 2039.450714
                    else:
                        return 1821.605789
