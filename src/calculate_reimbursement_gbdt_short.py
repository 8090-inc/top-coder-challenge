#!/usr/bin/env python3
"""Auto-generated GBDT scorer."""
from typing import Union
def calculate_reimbursement_gbdt_short(trip_duration_days: Union[int,float],
                    miles_traveled: Union[int,float],
                    total_receipts_amount: Union[int,float]) -> float:
        # --- BEGIN MAIN FUNCTION ---
        # Always initialize result to 0.0 and return round(result, 2) at the end.
        result = 0.0
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
        result += 0.05000000 * tree_0(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_1(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_2(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_3(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_4(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_5(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_6(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_7(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_8(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_9(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_10(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_11(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_12(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_13(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_14(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_15(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_16(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_17(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_18(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_19(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_20(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_21(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_22(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_23(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_24(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_25(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_26(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_27(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_28(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_29(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_30(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_31(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_32(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_33(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_34(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_35(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_36(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_37(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_38(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_39(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_40(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_41(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_42(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_43(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_44(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_45(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_46(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_47(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_48(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_49(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_50(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_51(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_52(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_53(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_54(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_55(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_56(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_57(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_58(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_59(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_60(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_61(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_62(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_63(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_64(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_65(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_66(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_67(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_68(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_69(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_70(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_71(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_72(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_73(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_74(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_75(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_76(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_77(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_78(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_79(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_80(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_81(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_82(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_83(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_84(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_85(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_86(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_87(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_88(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_89(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_90(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_91(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_92(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_93(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_94(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_95(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_96(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_97(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_98(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 0.05000000 * tree_99(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq)
        result += 1021.84282392
        return round(result, 2)

def tree_0(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 848.100006:
        if miles <= 574.000000:
            if receipts <= 541.885010:
                return -643.785847
            else:
                return -349.182199
        else:
            if receipts <= 563.100006:
                return -311.246538
            else:
                return -14.377824
    else:
        if receipts <= 1074.315002:
            if days_receipts <= 4125.880005:
                return 102.371462
            else:
                return -603.672824
        else:
            if miles_lt400 <= 70.500000:
                return -34.096824
            else:
                return 401.695868


def tree_1(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 844.244995:
        if miles_400_800 <= 174.000000:
            if days_receipts <= 665.450012:
                return -666.390541
            else:
                return -410.176532
        else:
            if receipts <= 559.670013:
                return -313.347283
            else:
                return -20.812933
    else:
        if receipts <= 1074.315002:
            if days_receipts <= 4125.880005:
                return 108.261103
            else:
                return -573.489183
        else:
            if miles <= 70.500000:
                return 38.017732
            else:
                return 387.748565


def tree_2(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 890.744995:
        if miles <= 593.000000:
            if receipts <= 562.035004:
                return -597.844626
            else:
                return -270.102707
        else:
            if rec_lt1500 <= 535.235016:
                return -277.869584
            else:
                return -24.466098
    else:
        if rec_lt1500 <= 1100.765015:
            if days_receipts <= 4094.520020:
                return 84.661220
            else:
                return -544.814724
        else:
            if days_receipts <= 3341.770020:
                return 272.237944
            else:
                return 405.265692


def tree_3(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 795.385010:
        if miles <= 583.000000:
            if receipts <= 541.885010:
                return -557.474813
            else:
                return -303.965711
        else:
            if receipts <= 591.994995:
                return -271.560983
            else:
                return -21.819981
    else:
        if receipts <= 1074.315002:
            if days_receipts <= 958.475006:
                return -166.414238
            else:
                return 122.989805
        else:
            if miles_lt400 <= 70.500000:
                return -14.696460
            else:
                return 340.894066


def tree_4(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 905.619995:
        if days_receipts <= 487.995010:
            if miles_400_800 <= 64.500000:
                return -609.219818
            else:
                return -379.083747
        else:
            if miles <= 649.089996:
                return -319.790380
            else:
                return -110.033937
    else:
        if miles_400_800 <= 221.000000:
            if days_receipts <= 4424.100098:
                return 106.576214
            else:
                return 297.855291
        else:
            if days_receipts <= 2271.224976:
                return 245.652359
            else:
                return 436.912873


def tree_5(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 716.859985:
        if miles_400_800 <= 183.000000:
            if days <= 2.500000:
                return -605.388139
            else:
                return -394.843567
        else:
            if days <= 2.500000:
                return -311.511739
            else:
                return -160.383928
    else:
        if rec_lt1500 <= 1072.804993:
            if miles_lt400 <= 394.050003:
                return -121.602248
            else:
                return 144.747528
        else:
            if miles <= 70.500000:
                return -14.995758
            else:
                return 318.470067


def tree_6(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 815.024994:
        if miles_400_800 <= 69.000000:
            if days_receipts <= 1416.894958:
                return -528.803248
            else:
                return -229.215133
        else:
            if days_receipts <= 1361.345032:
                return -263.275378
            else:
                return -47.790861
    else:
        if rec_lt1500 <= 1074.315002:
            if miles_gt800 <= 33.500000:
                return -74.967536
            else:
                return 210.272274
        else:
            if miles_400_800 <= 154.000000:
                return 207.541895
            else:
                return 344.308741


def tree_7(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 884.119995:
        if days_receipts <= 463.594986:
            if miles_400_800 <= 137.500000:
                return -528.435607
            else:
                return -323.635222
        else:
            if miles_400_800 <= 262.500000:
                return -306.828466
            else:
                return -116.541407
    else:
        if miles_400_800 <= 221.000000:
            if days_receipts <= 4424.100098:
                return 87.553951
            else:
                return 257.124220
        else:
            if days_receipts <= 2271.224976:
                return 201.195490
            else:
                return 373.007595


def tree_8(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 784.239990:
        if miles_400_800 <= 72.000000:
            if days_receipts <= 1380.630005:
                return -478.453737
            else:
                return -233.425020
        else:
            if rec_lt1500 <= 581.250000:
                return -239.009452
            else:
                return -30.796136
    else:
        if receipts <= 1074.315002:
            if days_receipts <= 4125.880005:
                return 68.181673
            else:
                return -523.601497
        else:
            if miles <= 70.500000:
                return -86.407650
            else:
                return 268.032356


def tree_9(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 823.764984:
        if miles <= 298.500000:
            if days_receipts <= 31.280001:
                return -567.896451
            else:
                return -386.452198
        else:
            if days_receipts <= 1416.605042:
                return -243.418133
            else:
                return -31.489880
    else:
        if days_receipts <= 2112.574951:
            if miles <= 460.000000:
                return -66.763653
            else:
                return 148.637293
        else:
            if miles <= 421.000000:
                return 168.962104
            else:
                return 317.107880


def tree_10(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 890.744995:
        if days_receipts <= 463.594986:
            if miles <= 464.500000:
                return -474.769293
            else:
                return -278.277523
        else:
            if miles_400_800 <= 262.500000:
                return -249.080208
            else:
                return -93.114807
    else:
        if miles <= 196.500000:
            if days_receipts <= 7235.729980:
                return 88.830377
            else:
                return -261.036833
        else:
            if days_receipts <= 2301.540039:
                return 130.466728
            else:
                return 297.157283


def tree_11(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 780.375000:
        if miles_lt400 <= 266.000000:
            if days <= 2.500000:
                return -494.255988
            else:
                return -316.682239
        else:
            if days <= 2.500000:
                return -273.913932
            else:
                return -113.020674
    else:
        if days_receipts <= 3363.699951:
            if miles_400_800 <= 228.000000:
                return 15.667720
            else:
                return 180.430010
        else:
            if miles_400_800 <= 339.000000:
                return 229.479322
            else:
                return 331.092714


def tree_12(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 665.750000:
        if miles <= 298.500000:
            if days <= 2.500000:
                return -453.411116
            else:
                return -314.263443
        else:
            if days <= 2.500000:
                return -269.473128
            else:
                return -119.704985
    else:
        if days_receipts <= 2148.010010:
            if miles_lt400 <= 388.550003:
                return -98.514567
            else:
                return 97.863764
        else:
            if miles <= 175.000000:
                return 54.888173
            else:
                return 250.513242


def tree_13(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 583.940002:
        if miles_400_800 <= 69.000000:
            if days <= 2.500000:
                return -438.715419
            else:
                return -275.135580
        else:
            if days <= 2.500000:
                return -234.471172
            else:
                return -122.646737
    else:
        if receipts <= 1074.315002:
            if miles_400_800 <= 132.000000:
                return -124.390530
            else:
                return 77.833188
        else:
            if miles_lt400 <= 70.500000:
                return -23.132217
            else:
                return 209.716838


def tree_14(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 774.895020:
        if miles_400_800 <= 74.000000:
            if days <= 2.500000:
                return -381.860691
            else:
                return -248.180490
        else:
            if days_receipts <= 445.674988:
                return -237.104602
            else:
                return -111.874814
    else:
        if days_receipts <= 1106.899963:
            if miles <= 532.000000:
                return -151.020849
            else:
                return 31.976278
        else:
            if miles <= 175.000000:
                return 31.865370
            else:
                return 219.244317


def tree_15(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 639.154999:
        if miles_400_800 <= 69.000000:
            if days_receipts <= 1591.919983:
                return -350.896873
            else:
                return -141.169135
        else:
            if miles_per_day <= 514.250000:
                return -127.457939
            else:
                return -245.984250
    else:
        if receipts <= 1074.315002:
            if miles_400_800 <= 211.000000:
                return -96.715800
            else:
                return 111.107241
        else:
            if miles_400_800 <= 215.000000:
                return 125.441605
            else:
                return 230.277991


def tree_16(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 859.579987:
        if days_receipts <= 463.594986:
            if miles_lt400 <= 144.000000:
                return -379.968638
            else:
                return -247.240530
        else:
            if miles <= 282.925003:
                return -210.454333
            else:
                return -86.144021
    else:
        if days_receipts <= 5098.540039:
            if miles <= 506.000000:
                return 25.529218
            else:
                return 153.873648
        else:
            if miles_400_800 <= 55.500000:
                return 207.552436
            else:
                return 294.102227


def tree_17(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 1070.479980:
        if rec_lt1500 <= 583.940002:
            if miles_400_800 <= 72.000000:
                return -291.605185
            else:
                return -136.292700
        else:
            if days_receipts <= 4040.939941:
                return -20.571467
            else:
                return -545.910380
    else:
        if days_receipts <= 2160.145020:
            if days_receipts <= 1692.094971:
                return 109.553254
            else:
                return -9.651078
        else:
            if miles <= 70.500000:
                return -65.025102
            else:
                return 200.648926


def tree_18(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 583.940002:
        if miles_400_800 <= 69.000000:
            if days <= 2.500000:
                return -349.054439
            else:
                return -222.089976
        else:
            if days <= 2.500000:
                return -183.355538
            else:
                return -107.281248
    else:
        if rec_lt1500 <= 1070.479980:
            if days_receipts <= 4125.880005:
                return -0.644509
            else:
                return -518.614861
        else:
            if days_receipts <= 5485.420166:
                return 123.469649
            else:
                return 239.527319


def tree_19(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 639.154999:
        if days_receipts <= 419.080002:
            if days <= 1.500000:
                return -329.888225
            else:
                return -206.092109
        else:
            if miles_400_800 <= 96.000000:
                return -186.115466
            else:
                return -91.463386
    else:
        if rec_lt1500 <= 1070.479980:
            if days_receipts <= 4094.520020:
                return 4.593410
            else:
                return -492.684118
        else:
            if miles_per_day <= 17.625000:
                return -123.577718
            else:
                return 156.614295


def tree_20(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 884.119995:
        if days_receipts <= 1081.229980:
            if miles <= 192.000000:
                return -311.072184
            else:
                return -179.617155
        else:
            if miles_400_800 <= 302.500000:
                return -101.598200
            else:
                return -2.524653
    else:
        if days_receipts <= 3860.830078:
            if miles <= 506.000000:
                return 9.653497
            else:
                return 115.621621
        else:
            if miles <= 76.500000:
                return -90.947441
            else:
                return 204.920456


def tree_21(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1377.275024:
        if miles_400_800 <= 183.000000:
            if days_receipts <= 26.415000:
                return -326.752560
            else:
                return -194.699665
        else:
            if receipts <= 726.470001:
                return -119.438177
            else:
                return 57.710854
    else:
        if rec_lt1500 <= 1254.604980:
            if miles_per_day <= 175.000000:
                return -33.977808
            else:
                return 77.843485
        else:
            if miles <= 596.000000:
                return 97.761172
            else:
                return 199.761691


def tree_22(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 652.279999:
        if days <= 1.500000:
            if miles_400_800 <= 166.500000:
                return -311.226529
            else:
                return -158.268041
        else:
            if miles <= 191.000000:
                return -209.985686
            else:
                return -97.662614
    else:
        if receipts <= 1074.315002:
            if days_receipts <= 4094.520020:
                return 9.823033
            else:
                return -476.597044
        else:
            if days <= 1.500000:
                return 56.439559
            else:
                return 159.662808


def tree_23(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 583.940002:
        if miles_400_800 <= 69.000000:
            if days <= 2.500000:
                return -279.193116
            else:
                return -173.654866
        else:
            if days <= 2.500000:
                return -153.833773
            else:
                return -79.247653
    else:
        if miles_400_800 <= 62.500000:
            if receipts <= 1228.675049:
                return -69.598151
            else:
                return 73.870498
        else:
            if days_receipts <= 1880.479980:
                return 46.248274
            else:
                return 182.924461


def tree_24(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 780.375000:
        if days_receipts <= 419.080002:
            if days <= 1.500000:
                return -239.924318
            else:
                return -160.230395
        else:
            if miles <= 662.500000:
                return -135.152939
            else:
                return -48.927733
    else:
        if miles <= 621.000000:
            if days_receipts <= 2799.539917:
                return -45.284864
            else:
                return 76.906813
        else:
            if days <= 3.500000:
                return 128.927707
            else:
                return 253.374706


def tree_25(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 890.744995:
        if miles_400_800 <= 312.500000:
            if days_receipts <= 48.135000:
                return -259.981890
            else:
                return -142.682881
        else:
            if days_receipts <= 1304.914978:
                return -93.896237
            else:
                return 51.922392
    else:
        if miles <= 759.000000:
            if days_receipts <= 9129.220215:
                return 70.736811
            else:
                return -313.063171
        else:
            if days_receipts <= 8104.740234:
                return 150.824742
            else:
                return 309.406361


def tree_26(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 639.154999:
        if miles_lt400 <= 156.500000:
            if days <= 1.500000:
                return -305.298061
            else:
                return -195.215414
        else:
            if days <= 2.500000:
                return -147.976791
            else:
                return -56.761845
    else:
        if days_receipts <= 3860.830078:
            if miles <= 623.500000:
                return -2.577850
            else:
                return 82.596628
        else:
            if miles_per_day <= 19.125000:
                return -167.762865
            else:
                return 170.045511


def tree_27(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1065.900024:
        if days_receipts <= 419.080002:
            if days <= 1.500000:
                return -230.931093
            else:
                return -130.586430
        else:
            if receipts <= 1029.195007:
                return -69.114993
            else:
                return -465.171741
    else:
        if miles_per_day <= 1070.000000:
            if miles_400_800 <= 154.000000:
                return 57.110123
            else:
                return 137.139412
        else:
            if miles <= 1097.500000:
                return -834.982781
            else:
                return 119.982003


def tree_28(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1911.695007:
        if rec_gt1500_sq <= 90573.687500:
            if rec_lt1500 <= 958.475006:
                return -117.420655
            else:
                return 61.118455
        else:
            return -793.233642
    else:
        if miles_lt400 <= 308.000000:
            if miles <= 70.500000:
                return -113.357866
            else:
                return 46.581608
        else:
            if days_receipts <= 7379.455078:
                return 113.075813
            else:
                return 204.381490


def tree_29(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 639.154999:
        if miles_400_800 <= 74.000000:
            if days <= 2.500000:
                return -235.730663
            else:
                return -121.137496
        else:
            if days <= 2.500000:
                return -101.511178
            else:
                return -37.701019
    else:
        if miles_400_800 <= 221.000000:
            if rec_lt1500 <= 1065.900024:
                return -68.707065
            else:
                return 47.135311
        else:
            if miles_per_day <= 1075.000000:
                return 118.962138
            else:
                return -102.069370


def tree_30(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 815.024994:
        if miles <= 144.000000:
            if miles_per_day <= 41.333332:
                return -148.280706
            else:
                return -250.233442
        else:
            if days <= 2.500000:
                return -125.350261
            else:
                return -38.253891
    else:
        if days_receipts <= 2745.379883:
            if miles <= 526.500000:
                return -48.232968
            else:
                return 54.463314
        else:
            if miles <= 70.500000:
                return -96.705709
            else:
                return 110.690460


def tree_31(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1364.140015:
        if miles_400_800 <= 74.000000:
            if receipts <= 652.339996:
                return -185.140902
            else:
                return -86.759109
        else:
            if rec_lt1500 <= 801.660004:
                return -72.326489
            else:
                return 25.791639
    else:
        if miles_lt400 <= 100.500000:
            if days_receipts <= 9129.220215:
                return -8.175223
            else:
                return -768.400937
        else:
            if days_receipts <= 3673.000000:
                return 35.555128
            else:
                return 121.662989


def tree_32(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 583.940002:
        if miles <= 474.000000:
            if days <= 2.500000:
                return -198.435741
            else:
                return -109.141242
        else:
            if miles_per_day <= 348.025009:
                return -31.530522
            else:
                return -94.507100
    else:
        if days_receipts <= 2590.915039:
            if miles <= 664.000000:
                return -45.824086
            else:
                return 46.727724
        else:
            if miles <= 70.500000:
                return -81.958841
            else:
                return 90.876164


def tree_33(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1364.140015:
        if miles_lt400 <= 191.000000:
            if days <= 1.500000:
                return -216.404437
            else:
                return -151.883844
        else:
            if miles <= 920.000000:
                return -93.866913
            else:
                return -24.567485
    else:
        if miles <= 756.500000:
            if receipts <= 1249.039978:
                return -14.560511
            else:
                return 64.156809
        else:
            if days_receipts <= 8104.740234:
                return 101.233757
            else:
                return 229.824087


def tree_34(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 890.744995:
        if days <= 1.500000:
            if miles_per_day <= 539.500000:
                return -171.859679
            else:
                return -78.291488
        else:
            if days_receipts <= 3348.479980:
                return -52.638650
            else:
                return -373.763622
    else:
        if days_receipts <= 9314.080078:
            if days_receipts <= 9144.000000:
                return 63.382945
            else:
                return -729.090789
        else:
            if miles_per_day <= 142.125000:
                return 159.160215
            else:
                return 251.191930


def tree_35(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1070.479980:
        if miles_400_800 <= 74.000000:
            if days_receipts <= 4094.520020:
                return -106.432098
            else:
                return -459.409484
        else:
            if receipts <= 652.279999:
                return -45.269837
            else:
                return 43.925689
    else:
        if days <= 3.500000:
            if miles <= 615.000000:
                return 25.268385
            else:
                return 83.007501
        else:
            if miles_400_800 <= 171.000000:
                return 112.654467
            else:
                return 176.726653


def tree_36(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1922.164978:
        if receipts <= 1753.284973:
            if miles <= 474.000000:
                return -113.113189
            else:
                return -26.172821
        else:
            return -767.687009
    else:
        if miles_400_800 <= 255.000000:
            if miles <= 643.000000:
                return 36.324531
            else:
                return -237.934187
        else:
            if days_receipts <= 6629.520020:
                return 90.911996
            else:
                return 179.610952


def tree_37(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1364.140015:
        if miles_400_800 <= 137.500000:
            if days <= 2.500000:
                return -133.165070
            else:
                return -65.714786
        else:
            if rec_lt1500 <= 924.565002:
                return -40.617080
            else:
                return 29.793484
    else:
        if rec_lt1500 <= 1136.040039:
            if miles_per_day <= 80.583332:
                return -80.704362
            else:
                return 35.273761
        else:
            if days <= 3.500000:
                return 60.974536
            else:
                return 126.263205


def tree_38(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 890.744995:
        if miles_gt800 <= 33.500000:
            if miles_gt800 <= 20.000000:
                return -80.667086
            else:
                return -363.581014
        else:
            if days_receipts <= 1626.244995:
                return -30.776606
            else:
                return 71.888736
    else:
        if miles_400_800 <= 219.000000:
            if days_receipts <= 4839.974854:
                return -5.253897
            else:
                return 54.276377
        else:
            if days <= 3.500000:
                return 70.491124
            else:
                return 156.325639


def tree_39(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 562.035004:
        if miles_400_800 <= 183.000000:
            if days <= 2.500000:
                return -149.220338
            else:
                return -78.432073
        else:
            if miles_per_day <= 689.500000:
                return -28.088722
            else:
                return -89.535081
    else:
        if days_receipts <= 4276.719971:
            if days_receipts <= 4243.909912:
                return 21.739881
            else:
                return -433.957323
        else:
            if miles_per_day <= 19.875000:
                return -68.403402
            else:
                return 97.778038


def tree_40(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1922.164978:
        if rec_gt1500_sq <= 90573.687500:
            if miles_lt400 <= 190.500000:
                return -131.169742
            else:
                return -25.231443
        else:
            return -736.962936
    else:
        if miles <= 655.000000:
            if receipts <= 1065.900024:
                return -78.609008
            else:
                return 37.822720
        else:
            if days_receipts <= 8104.740234:
                return 85.416813
            else:
                return 203.265956


def tree_41(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 639.154999:
        if miles <= 472.000000:
            if miles_400_800 <= 41.500000:
                return -103.743917
            else:
                return -221.365650
        else:
            if receipts <= 615.279999:
                return -40.726669
            else:
                return -220.924355
    else:
        if miles <= 621.000000:
            if days_receipts <= 4839.974854:
                return -8.247630
            else:
                return 54.620428
        else:
            if miles <= 1184.500000:
                return 65.783662
            else:
                return 187.304332


def tree_42(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 639.154999:
        if miles_lt400 <= 176.000000:
            if days <= 1.500000:
                return -162.148810
            else:
                return -94.379008
        else:
            if days <= 2.500000:
                return -73.529790
            else:
                return -17.793392
    else:
        if miles_per_day <= 17.625000:
            if days_receipts <= 7146.439941:
                return 82.855284
            else:
                return -710.314166
        else:
            if days_receipts <= 5322.090088:
                return 21.300693
            else:
                return 93.324917


def tree_43(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1988.119995:
        if rec_gt1500_sq <= 90573.687500:
            if miles <= 472.000000:
                return -78.407818
            else:
                return -7.510077
        else:
            if receipts <= 1887.224976:
                return -704.469007
            else:
                return -0.664696
    else:
        if miles_lt400 <= 308.000000:
            if miles_lt400 <= 283.000000:
                return 25.016775
            else:
                return -133.388092
        else:
            if days_receipts <= 7302.074951:
                return 52.355129
            else:
                return 122.261305


def tree_44(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 639.154999:
        if days <= 1.500000:
            if miles <= 539.500000:
                return -124.897491
            else:
                return -61.282229
        else:
            if miles_lt400 <= 156.500000:
                return -90.286238
            else:
                return -28.856610
    else:
        if miles_per_day <= 17.625000:
            if miles <= 43.500000:
                return 49.470812
            else:
                return -676.049296
        else:
            if days_receipts <= 7347.955078:
                return 24.563304
            else:
                return 114.896789


def tree_45(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 890.744995:
        if days_receipts <= 3348.479980:
            if days <= 1.500000:
                return -95.788221
            else:
                return -30.012646
        else:
            return -358.959937
    else:
        if miles_lt400 <= 313.000000:
            if miles_lt400 <= 283.000000:
                return 9.075972
            else:
                return -88.323168
        else:
            if days <= 3.500000:
                return 36.587165
            else:
                return 113.808685


def tree_46(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1922.164978:
        if rec_gt1500_sq <= 90573.687500:
            if rec_lt1500 <= 1348.290039:
                return -53.379178
            else:
                return 52.033317
        else:
            return -672.303080
    else:
        if miles_per_day <= 19.125000:
            if miles <= 49.000000:
                return 76.590429
            else:
                return -642.700630
        else:
            if days_receipts <= 7318.169922:
                return 37.056961
            else:
                return 107.625291


def tree_47(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1361.345032:
        if miles_lt400 <= 191.000000:
            if miles <= 86.500000:
                return -126.848067
            else:
                return -75.572405
        else:
            if receipts <= 217.614998:
                return 6.295440
            else:
                return -49.387380
    else:
        if miles <= 309.000000:
            if days_receipts <= 9129.220215:
                return 3.335449
            else:
                return -261.405433
        else:
            if miles_per_day <= 1075.000000:
                return 58.200538
            else:
                return -102.495898


def tree_48(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 1136.040039:
        if days_receipts <= 4094.520020:
            if miles_lt400 <= 156.000000:
                return -86.739139
            else:
                return -17.688567
        else:
            return -401.143882
    else:
        if miles_per_day <= 17.625000:
            if miles_lt400 <= 49.000000:
                return 101.613048
            else:
                return -597.495327
        else:
            if days_receipts <= 7464.574951:
                return 25.679523
            else:
                return 108.812557


def tree_49(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 639.154999:
        if days <= 1.500000:
            if miles <= 539.500000:
                return -113.169453
            else:
                return -54.045014
        else:
            if receipts <= 230.700005:
                return -5.060445
            else:
                return -60.652717
    else:
        if miles_400_800 <= 222.000000:
            if days_receipts <= 4424.100098:
                return -17.175222
            else:
                return 37.361589
        else:
            if miles_gt800 <= 388.000000:
                return 55.470846
            else:
                return 163.308971


def tree_50(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 562.035004:
        if days <= 1.500000:
            if receipts <= 528.580002:
                return -83.095049
            else:
                return -294.671014
        else:
            if miles <= 176.000000:
                return -73.330425
            else:
                return -11.710371
    else:
        if miles_per_day <= 1075.000000:
            if miles_400_800 <= 255.000000:
                return 6.188510
            else:
                return 60.519086
        else:
            if rec_gt1500_sq <= 48540.031250:
                return 65.471437
            else:
                return -637.620650


def tree_51(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1067.114990:
        if days_receipts <= 4094.520020:
            if miles_lt400 <= 144.000000:
                return -89.558092
            else:
                return -23.823293
        else:
            return -380.537353
    else:
        if miles <= 614.000000:
            if days_receipts <= 9362.259766:
                return 7.211103
            else:
                return 119.824867
        else:
            if miles <= 1184.500000:
                return 51.798731
            else:
                return 154.253716


def tree_52(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1374.860046:
        if miles_lt400 <= 192.000000:
            if days_receipts <= 8.325000:
                return -140.562821
            else:
                return -75.931826
        else:
            if miles_per_day <= 343.396667:
                return -6.201733
            else:
                return -44.728126
    else:
        if miles <= 82.000000:
            if days_receipts <= 7168.439941:
                return -13.790059
            else:
                return -570.158621
        else:
            if miles_per_day <= 1070.000000:
                return 29.077419
            else:
                return -99.290340


def tree_53(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 958.579987:
        if days <= 1.500000:
            if miles_lt400 <= 196.000000:
                return -109.698572
            else:
                return -58.164713
        else:
            if miles_lt400 <= 156.500000:
                return -57.893958
            else:
                return -10.571893
    else:
        if miles_per_day <= 17.625000:
            if days <= 3.500000:
                return 94.683839
            else:
                return -541.650690
        else:
            if days <= 3.500000:
                return 17.773836
            else:
                return 88.321727


def tree_54(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 1127.369995:
        if days_receipts <= 4040.939941:
            if miles <= 144.000000:
                return -74.372359
            else:
                return -18.623432
        else:
            return -367.380442
    else:
        if miles <= 1188.000000:
            if days_receipts <= 3155.349976:
                return 1.319065
            else:
                return 39.778165
        else:
            if receipts <= 1707.625000:
                return 228.085858
            else:
                return 123.255161


def tree_55(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1364.140015:
        if miles <= 833.500000:
            if days <= 2.500000:
                return -66.836914
            else:
                return -31.976779
        else:
            if miles_per_day <= 331.426666:
                return 41.282687
            else:
                return -5.402614
    else:
        if days_receipts <= 7823.920166:
            if rec_lt1500 <= 890.744995:
                return -8.095195
            else:
                return 27.759765
        else:
            if rec_gt1500 <= 746.815002:
                return 58.870578
            else:
                return 107.286156


def tree_56(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1922.164978:
        if rec_gt1500_sq <= 90573.687500:
            if miles_lt400 <= 191.000000:
                return -60.684299
            else:
                return -9.043319
        else:
            return -605.707670
    else:
        if miles_400_800 <= 268.500000:
            if miles_per_day <= 17.625000:
                return -86.152516
            else:
                return 12.554432
        else:
            if miles_per_day <= 312.750000:
                return 81.874773
            else:
                return 32.699813


def tree_57(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 639.154999:
        if miles_gt800 <= 55.400000:
            if days_receipts <= 2376.140015:
                return -47.599629
            else:
                return -208.843900
        else:
            if miles_per_day <= 350.166672:
                return 24.676201
            else:
                return -28.129082
    else:
        if miles <= 621.000000:
            if days_receipts <= 4839.974854:
                return -16.241013
            else:
                return 28.391768
        else:
            if days <= 3.500000:
                return 35.843086
            else:
                return 82.020676


def tree_58(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 1161.380005:
        if miles_gt800 <= 33.500000:
            if miles_gt800 <= 20.000000:
                return -28.958118
            else:
                return -357.282836
        else:
            if miles_per_day <= 301.300003:
                return 57.943928
            else:
                return -5.594399
    else:
        if miles_400_800 <= 214.000000:
            if days_receipts <= 4414.050049:
                return -11.712091
            else:
                return 30.718018
        else:
            if receipts <= 1178.070007:
                return 211.866432
            else:
                return 50.001349


def tree_59(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 562.035004:
        if receipts <= 549.614990:
            if miles_lt400 <= 78.000000:
                return -85.389072
            else:
                return -28.184510
        else:
            return -265.048586
    else:
        if miles <= 655.000000:
            if days_receipts <= 2745.379883:
                return -29.492812
            else:
                return 18.191931
        else:
            if miles_per_day <= 198.625000:
                return 133.020318
            else:
                return 27.246566


def tree_60(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1364.140015:
        if miles_400_800 <= 183.000000:
            if miles_per_day <= 443.500000:
                return -39.551686
            else:
                return -124.077608
        else:
            if rec_lt1500 <= 230.700005:
                return 18.650996
            else:
                return -10.823588
    else:
        if miles <= 82.000000:
            if days_receipts <= 7168.439941:
                return 10.766875
            else:
                return -521.478831
        else:
            if days_receipts <= 7823.920166:
                return 18.063270
            else:
                return 83.183358


def tree_61(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1161.380005:
        if days_receipts <= 4125.880005:
            if miles_lt400 <= 144.000000:
                return -57.038257
            else:
                return -8.716681
        else:
            return -350.579933
    else:
        if miles_per_day <= 1075.000000:
            if miles_per_day <= 17.625000:
                return -101.986003
            else:
                return 30.042383
        else:
            if miles <= 1093.500000:
                return -581.980000
            else:
                return 78.977364


def tree_62(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 632.959991:
        if days <= 1.500000:
            if miles_400_800 <= 139.500000:
                return -78.929183
            else:
                return -18.360199
        else:
            if rec_lt1500 <= 615.279999:
                return -11.029421
            else:
                return -195.946488
    else:
        if miles_per_day <= 17.625000:
            if days <= 3.500000:
                return 91.576246
            else:
                return -490.305590
        else:
            if days_receipts <= 7823.920166:
                return 14.542054
            else:
                return 68.099616


def tree_63(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1922.164978:
        if receipts <= 1800.830017:
            if miles_400_800 <= 74.000000:
                return -44.566482
            else:
                return 8.510981
        else:
            return -553.608103
    else:
        if days_receipts <= 9353.660156:
            if days_receipts <= 9144.000000:
                return 24.526900
            else:
                return -465.790310
        else:
            if miles_per_day <= 145.125000:
                return 80.396876
            else:
                return 109.608748


def tree_64(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1922.164978:
        if rec_gt1500_sq <= 75401.228516:
            if rec_lt1500 <= 1377.275024:
                return -23.879825
            else:
                return 38.435821
        else:
            return -525.927698
    else:
        if miles <= 1188.000000:
            if miles <= 36.000000:
                return 139.639629
            else:
                return 24.266028
        else:
            if days <= 3.000000:
                return 195.552052
            else:
                return 107.538377


def tree_65(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 3673.000000:
        if miles_lt400 <= 92.000000:
            if rec_lt1500 <= 1235.465027:
                return -61.490079
            else:
                return 6.115873
        else:
            if miles_gt800 <= 294.000000:
                return -8.557066
            else:
                return 44.327763
    else:
        if days_receipts <= 9314.080078:
            if days_receipts <= 9144.000000:
                return 34.843687
            else:
                return -443.714096
        else:
            if miles_gt800 <= 52.000000:
                return 63.289987
            else:
                return 112.558327


def tree_66(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1257.164978:
        if miles_gt800 <= 33.500000:
            if miles_400_800 <= 398.500000:
                return -21.503263
            else:
                return -343.987248
        else:
            if days_receipts <= 507.815018:
                return -31.503094
            else:
                return 22.740417
    else:
        if days <= 3.500000:
            if miles_per_day <= 1038.000000:
                return 12.276063
            else:
                return 75.194105
        else:
            if miles <= 455.500000:
                return 40.323530
            else:
                return 78.612608


def tree_67(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 2170.800049:
        if rec_gt1500_sq <= 90573.687500:
            if rec_lt1500 <= 1410.429993:
                return -16.683700
            else:
                return 39.734378
        else:
            if days_receipts <= 1887.224976:
                return -502.963165
            else:
                return -11.901679
    else:
        if days_receipts <= 9314.080078:
            if days_receipts <= 9144.000000:
                return 21.497781
            else:
                return -423.544568
        else:
            if miles_gt800 <= 52.000000:
                return 59.653135
            else:
                return 102.999780


def tree_68(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 562.035004:
        if rec_lt1500 <= 550.529999:
            if miles_400_800 <= 183.000000:
                return -42.150775
            else:
                return -4.122036
        else:
            return -235.450466
    else:
        if miles_gt800 <= 364.000000:
            if days_receipts <= 9314.080078:
                return 4.382431
            else:
                return 68.548817
        else:
            if receipts <= 1294.214966:
                return 158.420108
            else:
                return 48.896580


def tree_69(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1396.914978:
        if miles_400_800 <= 183.000000:
            if miles_per_day <= 443.500000:
                return -28.388249
            else:
                return -92.431398
        else:
            if days_receipts <= 418.659988:
                return -38.536726
            else:
                return 4.367674
    else:
        if miles <= 1188.000000:
            if miles_per_day <= 1075.000000:
                return 16.927433
            else:
                return -46.424032
        else:
            if miles_gt800 <= 390.000000:
                return 173.425146
            else:
                return 86.467327


def tree_70(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1922.164978:
        if receipts <= 1800.830017:
            if miles_400_800 <= 183.000000:
                return -34.622180
            else:
                return 2.489160
        else:
            return -475.712926
    else:
        if miles_per_day <= 457.500000:
            if days_receipts <= 9314.080078:
                return 4.383658
            else:
                return 70.044943
        else:
            if miles <= 1015.500000:
                return 31.204760
            else:
                return 99.298958


def tree_71(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1596.179993:
        if miles_lt400 <= 86.500000:
            if receipts <= 20.545000:
                return -78.069177
            else:
                return -26.670271
        else:
            if receipts <= 217.614998:
                return 12.849345
            else:
                return -20.976375
    else:
        if miles_per_day <= 1075.000000:
            if miles <= 949.500000:
                return 10.184100
            else:
                return 52.530541
        else:
            if days_receipts <= 1910.464966:
                return -451.927280
            else:
                return 90.394463


def tree_72(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 780.375000:
        if days <= 2.500000:
            if miles <= 920.000000:
                return -43.566317
            else:
                return 16.566288
        else:
            if receipts <= 270.125000:
                return 24.052264
            else:
                return -14.466680
    else:
        if miles <= 1188.000000:
            if days_receipts <= 7823.920166:
                return 10.814057
            else:
                return 54.185248
        else:
            if days_receipts <= 3163.640015:
                return 157.162413
            else:
                return 79.298251


def tree_73(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1215.919983:
        if miles <= 883.500000:
            if days <= 2.500000:
                return -36.975567
            else:
                return -5.651813
        else:
            if miles <= 900.500000:
                return 101.475414
            else:
                return -1.958442
    else:
        if miles_gt800 <= 388.000000:
            if miles_lt400 <= 292.500000:
                return -13.532073
            else:
                return 14.785165
        else:
            if days <= 3.000000:
                return 149.304293
            else:
                return 75.333338


def tree_74(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1383.275024:
        if days <= 2.500000:
            if miles_gt800 <= 120.000000:
                return -35.519317
            else:
                return 12.555670
        else:
            if receipts <= 230.700005:
                return 29.242622
            else:
                return -33.896054
    else:
        if miles <= 128.000000:
            if days_receipts <= 8223.119873:
                return 2.488869
            else:
                return -406.193879
        else:
            if miles_per_day <= 1075.000000:
                return 20.151287
            else:
                return -37.448707


def tree_75(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 443.425003:
        if rec_lt1500 <= 210.709999:
            if days <= 1.500000:
                return -48.288650
            else:
                return 14.188647
        else:
            if miles <= 1083.000000:
                return -53.852981
            else:
                return 7.796647
    else:
        if miles <= 1188.000000:
            if days_receipts <= 9314.080078:
                return 1.764567
            else:
                return 62.179560
        else:
            if miles_per_day <= 447.500000:
                return 70.559107
            else:
                return 140.831514


def tree_76(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 1.500000:
        if miles_gt800 <= 298.500000:
            if miles_per_day <= 1075.000000:
                return -17.100147
            else:
                return -219.286062
        else:
            if rec_gt1500 <= 255.720001:
                return 44.460121
            else:
                return 86.378986
    else:
        if miles_per_day <= 474.750000:
            if miles_per_day <= 358.000000:
                return 8.340417
            else:
                return -29.669772
        else:
            if miles_per_day <= 583.250000:
                return 54.748820
            else:
                return 121.599948


def tree_77(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles <= 183.000000:
        if days_receipts <= 8286.314941:
            if days_receipts <= 2890.229980:
                return -33.288699
            else:
                return 19.445616
        else:
            if rec_gt1500 <= 892.704987:
                return -386.389434
            else:
                return 87.656071
    else:
        if days_receipts <= 5308.545166:
            if miles_gt800 <= 294.000000:
                return -4.454986
            else:
                return 35.059409
        else:
            if days_receipts <= 9108.580078:
                return 26.922269
            else:
                return 70.540285


def tree_78(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 1161.380005:
        if days_receipts <= 4125.880005:
            if days_receipts <= 3641.979980:
                return -12.549697
            else:
                return 85.742729
        else:
            if days_receipts <= 4276.719971:
                return -340.907641
            else:
                return 54.540532
    else:
        if days <= 3.500000:
            if receipts <= 1188.210022:
                return 125.956970
            else:
                return 3.982296
        else:
            if miles_lt400 <= 47.000000:
                return 147.357985
            else:
                return 36.890858


def tree_79(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 2173.770020:
        if receipts <= 1800.830017:
            if miles <= 583.000000:
                return -24.567510
            else:
                return 4.404814
        else:
            if miles_gt800 <= 275.000000:
                return -3.931777
            else:
                return -417.838733
    else:
        if miles_per_day <= 17.625000:
            if days_receipts <= 7146.439941:
                return 71.899163
            else:
                return -368.914506
        else:
            if miles <= 1188.000000:
                return 13.428040
            else:
                return 76.447689


def tree_80(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 3.500000:
        if receipts <= 564.544983:
            if rec_lt1500 <= 549.614990:
                return -19.797644
            else:
                return -207.627997
        else:
            if miles_per_day <= 1093.500000:
                return 1.429555
            else:
                return 55.630061
    else:
        if rec_lt1500 <= 1069.179993:
            if rec_lt1500 <= 1031.470001:
                return 19.074807
            else:
                return -324.533661
        else:
            if miles_lt400 <= 41.000000:
                return 136.395127
            else:
                return 32.934122


def tree_81(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 443.425003:
        if receipts <= 223.795006:
            if days <= 1.500000:
                return -32.974379
            else:
                return 14.686992
        else:
            if miles <= 787.160004:
                return -50.424713
            else:
                return -11.871230
    else:
        if rec_lt1500 <= 451.449997:
            return 137.509648
        else:
            if miles <= 140.500000:
                return -20.343131
            else:
                return 9.344207


def tree_82(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 1159.470032:
        if days_receipts <= 4125.880005:
            if days_receipts <= 3641.979980:
                return -10.950642
            else:
                return 78.312912
        else:
            if miles_400_800 <= 200.000000:
                return -308.774188
            else:
                return 45.877205
    else:
        if miles_per_day <= 1075.000000:
            if miles <= 254.000000:
                return -11.395013
            else:
                return 18.547664
        else:
            if rec_gt1500_sq <= 48540.031250:
                return 47.064349
            else:
                return -397.485484


def tree_83(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles <= 613.000000:
        if days <= 2.500000:
            if rec_lt1500 <= 423.159988:
                return -40.621307
            else:
                return -14.979948
        else:
            if days_receipts <= 9129.220215:
                return 8.409444
            else:
                return -86.208559
    else:
        if rec_lt1500 <= 705.369995:
            if days_receipts <= 2416.780029:
                return -2.283907
            else:
                return -192.177888
        else:
            if miles_gt800 <= 275.000000:
                return 31.790186
            else:
                return -8.226733


def tree_84(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles_per_day <= 1017.000000:
        if days <= 1.500000:
            if rec_lt1500 <= 431.909988:
                return -43.317339
            else:
                return -5.460820
        else:
            if miles_per_day <= 573.500000:
                return 4.162973
            else:
                return 82.906051
    else:
        if receipts <= 713.900009:
            if rec_lt1500 <= 446.110001:
                return 4.094014
            else:
                return -16.833177
        else:
            if days_receipts <= 2011.359985:
                return 52.674625
            else:
                return 118.036786


def tree_85(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 9314.080078:
        if days_receipts <= 9144.000000:
            if days <= 3.500000:
                return -5.906965
            else:
                return 18.457575
        else:
            return -346.426300
    else:
        if receipts <= 2464.104980:
            if rec_gt1500_sq <= 714875.156250:
                return 89.987161
            else:
                return 59.239537
        else:
            if rec_gt1500_sq <= 934466.250000:
                return -21.432042
            else:
                return 60.253730


def tree_86(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles_lt400 <= 82.000000:
        if days_receipts <= 7377.560059:
            if days_receipts <= 3572.414917:
                return -32.204087
            else:
                return 75.480157
        else:
            return -329.104985
    else:
        if days <= 3.500000:
            if miles_gt800 <= 271.000000:
                return 1.201607
            else:
                return -38.242195
        else:
            if receipts <= 837.119995:
                return 49.401258
            else:
                return 16.032812


def tree_87(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles <= 304.000000:
        if miles_lt400 <= 283.000000:
            if miles <= 83.500000:
                return -29.636078
            else:
                return -0.708588
        else:
            if days <= 3.500000:
                return -10.298387
            else:
                return -295.688619
    else:
        if miles_per_day <= 116.625000:
            if miles <= 306.000000:
                return 132.409278
            else:
                return 48.484352
        else:
            if miles <= 508.000000:
                return -19.669344
            else:
                return 10.113219


def tree_88(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 3.500000:
        if miles <= 508.000000:
            if miles_per_day <= 443.500000:
                return -16.696435
            else:
                return -57.516589
        else:
            if miles_gt800 <= 259.574997:
                return 9.588988
            else:
                return -19.464046
    else:
        if miles_lt400 <= 295.500000:
            if miles_per_day <= 68.625000:
                return 19.324623
            else:
                return -280.904188
        else:
            if miles_lt400 <= 311.000000:
                return 125.788814
            else:
                return 38.944453


def tree_89(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 1065.900024:
        if days_receipts <= 4094.520020:
            if receipts <= 230.700005:
                return 7.054647
            else:
                return -17.763133
        else:
            return -266.858979
    else:
        if days_receipts <= 9314.080078:
            if days_receipts <= 9129.220215:
                return 10.449257
            else:
                return -312.134163
        else:
            if days_receipts <= 9381.780273:
                return 82.233279
            else:
                return 56.700244


def tree_90(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 9314.080078:
        if days_receipts <= 9144.000000:
            if days_receipts <= 1215.919983:
                return -13.329413
            else:
                return 6.754666
        else:
            return -296.527455
    else:
        if rec_gt1500_sq <= 714875.156250:
            return 78.121615
        else:
            if miles <= 133.500000:
                return 77.270371
            else:
                return 49.184204


def tree_91(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 430.324997:
        if rec_lt1500 <= 210.709999:
            if days <= 2.500000:
                return -20.616628
            else:
                return 28.796445
        else:
            if miles <= 217.375000:
                return -62.954834
            else:
                return -25.666455
    else:
        if miles_per_day <= 474.750000:
            if miles_per_day <= 358.000000:
                return 6.306332
            else:
                return -20.720279
        else:
            if miles_gt800 <= 290.000000:
                return 12.002462
            else:
                return 72.152946


def tree_92(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 3673.000000:
        if miles <= 1102.500000:
            if miles_per_day <= 1075.000000:
                return -4.159140
            else:
                return -184.608331
        else:
            if days_receipts <= 1584.200012:
                return 21.925819
            else:
                return 109.750945
    else:
        if miles_per_day <= 23.750000:
            if rec_gt1500_sq <= 284938.217773:
                return 111.552648
            else:
                return 48.865844
        else:
            if miles <= 596.000000:
                return 1.568148
            else:
                return 23.379634


def tree_93(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 7464.574951:
        if miles_per_day <= 486.750000:
            if miles_per_day <= 357.250000:
                return 1.907966
            else:
                return -30.839814
        else:
            if miles <= 1108.500000:
                return 6.786386
            else:
                return 58.935425
    else:
        if miles_per_day <= 279.625000:
            if miles_per_day <= 213.000000:
                return 27.693835
            else:
                return 60.097341
        else:
            return -41.211978


def tree_94(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles <= 10.500000:
        return 119.503800
    else:
        if miles_lt400 <= 140.500000:
            if days_receipts <= 9129.220215:
                return -15.065567
            else:
                return -285.844383
        else:
            if days_receipts <= 7823.920166:
                return -1.360906
            else:
                return 30.740563


def tree_95(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles_lt400 <= 78.000000:
        if days_receipts <= 7377.560059:
            if receipts <= 913.414978:
                return -40.519080
            else:
                return 23.352934
        else:
            return -271.552163
    else:
        if days_receipts <= 8840.600098:
            if miles_per_day <= 1098.500000:
                return -1.806093
            else:
                return 43.809025
        else:
            if rec_gt1500_sq <= 929498.468750:
                return 49.554143
            else:
                return 6.791865


def tree_96(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles <= 949.500000:
        if days <= 2.500000:
            if miles <= 537.500000:
                return -23.156268
            else:
                return -1.452269
        else:
            if receipts <= 270.125000:
                return 35.077324
            else:
                return -0.222232
    else:
        if days <= 2.500000:
            if days_receipts <= 1568.520020:
                return 19.132785
            else:
                return 60.946230
        else:
            if miles_per_day <= 348.025009:
                return 24.478648
            else:
                return -40.633403


def tree_97(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 1.500000:
        if miles_gt800 <= 287.000000:
            if miles_gt800 <= 275.000000:
                return -8.890544
            else:
                return -372.616789
        else:
            if rec_gt1500_sq <= 131433.437500:
                return 29.570082
            else:
                return 96.738066
    else:
        if miles_per_day <= 474.750000:
            if miles_per_day <= 350.166672:
                return 5.441605
            else:
                return -20.833039
        else:
            if miles_gt800 <= 366.500000:
                return 40.153101
            else:
                return 98.132087


def tree_98(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 3.500000:
        if miles <= 567.500000:
            if miles <= 560.000000:
                return -16.490281
            else:
                return -142.988021
        else:
            if miles_per_day <= 1038.000000:
                return 4.459945
            else:
                return 37.021537
    else:
        if miles_400_800 <= 324.500000:
            if miles <= 638.500000:
                return 21.214320
            else:
                return -45.125977
        else:
            if miles_per_day <= 196.750000:
                return 116.444865
            else:
                return 21.465323


def tree_99(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles_400_800 <= 137.500000:
        if miles_per_day <= 443.500000:
            if days_receipts <= 9362.259766:
                return -8.543031
            else:
                return 49.299827
        else:
            if miles <= 454.500000:
                return -185.123879
            else:
                return -52.948454
    else:
        if miles_per_day <= 165.250000:
            if rec_lt1500 <= 788.234985:
                return -186.121622
            else:
                return 40.505403
        else:
            if miles_per_day <= 196.750000:
                return 47.394418
            else:
                return 6.208413


if __name__ == "__main__":
    # Test with a sample input
    result = calculate_reimbursement_gbdt_short(trip_duration_days=5, miles_traveled=400, total_receipts_amount=1000)
    print(f"Test result: {result}")

