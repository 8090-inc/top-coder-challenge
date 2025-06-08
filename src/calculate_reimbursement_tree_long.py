#!/usr/bin/env python3
"""Auto-generated decision tree scorer."""
from typing import Union

def calculate_reimbursement_tree_long(trip_duration_days: Union[int,float],
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
    if rec_lt1500 <= 822.690002:
        if miles_400_800 <= 85.000000:
            if rec_lt1500 <= 581.829987:
                if miles <= 90.500000:
                    return 757.828000
                else:
                    if days <= 11.500000:
                        return 859.673529
                    else:
                        return 986.790000
            else:
                return 1163.152308
        else:
            if days_receipts <= 6201.945068:
                if miles_gt800 <= 88.500000:
                    if days <= 10.500000:
                        return 1058.710000
                    else:
                        return 1196.119000
                else:
                    if miles <= 1090.000000:
                        return 1294.966364
                    else:
                        return 1347.975000
            else:
                return 1500.532727
    else:
        if miles <= 503.000000:
            if days_receipts <= 13199.189941:
                if receipts <= 989.634979:
                    return 1283.687273
                else:
                    return 1497.586471
            else:
                if days <= 12.500000:
                    if rec_gt1500_sq <= 220460.250000:
                        if miles_per_day <= 28.511364:
                            return 1635.549231
                        else:
                            return 1733.192000
                    else:
                        if days_receipts <= 22425.149414:
                            return 1582.568000
                        else:
                            return 1636.006667
                else:
                    return 1750.761667
        else:
            if miles_gt800 <= 195.000000:
                if days <= 12.500000:
                    if miles <= 740.500000:
                        if days <= 10.500000:
                            return 1621.320714
                        else:
                            if days <= 11.500000:
                                return 1738.236250
                            else:
                                return 1806.455455
                    else:
                        if days_receipts <= 10846.859863:
                            return 1754.398000
                        else:
                            if rec_lt1500 <= 1491.695007:
                                return 1914.471429
                            else:
                                if rec_gt1500_sq <= 212024.171875:
                                    return 1829.059286
                                else:
                                    return 1789.078750
                else:
                    if rec_gt1500 <= 5.110000:
                        return 1851.192727
                    else:
                        if days_receipts <= 27853.700195:
                            return 1966.428000
                        else:
                            return 1911.777000
            else:
                if rec_gt1500_sq <= 83246.851562:
                    if miles <= 1040.000000:
                        return 2006.126364
                    else:
                        return 2115.700769
                else:
                    if days_receipts <= 22612.955078:
                        return 1833.765455
                    else:
                        return 1921.269286
