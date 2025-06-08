#!/usr/bin/env python3
"""Auto-generated GBDT scorer."""
from typing import Union
def calculate_reimbursement_gbdt_mid(trip_duration_days: Union[int,float],
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
        result += 1385.40792570
        return round(result, 2)

def tree_0(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 724.209991:
        if miles_400_800 <= 225.000000:
            if miles <= 262.970001:
                return -745.266973
            else:
                return -520.351140
        else:
            if days_receipts <= 3202.544922:
                return -363.210100
            else:
                return -60.181259
    else:
        if miles_400_800 <= 241.000000:
            if days_receipts <= 6232.725098:
                return -195.662926
            else:
                return 125.787320
        else:
            if days_receipts <= 6072.410156:
                return 135.334217
            else:
                return 411.772854


def tree_1(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 4343.149902:
        if miles <= 347.500000:
            if days_receipts <= 3251.750000:
                return -718.173383
            else:
                return -417.321076
        else:
            if receipts <= 500.494995:
                return -411.335934
            else:
                return -172.271665
    else:
        if miles_per_day <= 103.412498:
            if receipts <= 992.574982:
                return -214.032107
            else:
                return 128.519164
        else:
            if days_receipts <= 6072.410156:
                return 135.646490
            else:
                return 411.433271


def tree_2(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 4343.149902:
        if miles_lt400 <= 369.500000:
            if days_receipts <= 3896.299927:
                return -669.106688
            else:
                return -300.223624
        else:
            if miles <= 759.000000:
                return -389.402000
            else:
                return -175.199016
    else:
        if miles_per_day <= 103.512501:
            if days_receipts <= 6179.500000:
                return -178.831127
            else:
                return 111.864905
        else:
            if days_receipts <= 5601.875000:
                return 85.657881
            else:
                return 381.132231


def tree_3(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 839.369995:
        if miles_400_800 <= 225.000000:
            if miles_lt400 <= 262.970001:
                return -646.823129
            else:
                return -424.056124
        else:
            if days_receipts <= 3023.425049:
                return -336.006829
            else:
                return -22.603476
    else:
        if miles_per_day <= 103.412498:
            if days_receipts <= 6179.500000:
                return -146.860030
            else:
                return 104.531328
        else:
            if days <= 6.500000:
                return 254.062148
            else:
                return 472.782621


def tree_4(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 4343.074951:
        if miles_lt400 <= 358.000000:
            if days_receipts <= 4064.314941:
                return -597.893535
            else:
                return -265.370364
        else:
            if rec_lt1500 <= 500.494995:
                return -367.509055
            else:
                return -138.098691
    else:
        if miles_400_800 <= 231.500000:
            if receipts <= 998.095001:
                return -204.126499
            else:
                return 112.131354
        else:
            if days_receipts <= 6072.410156:
                return 107.263638
            else:
                return 344.093381


def tree_5(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 4241.949951:
        if miles_400_800 <= 228.500000:
            if miles <= 267.000000:
                return -586.383345
            else:
                return -432.033876
        else:
            if days_receipts <= 3202.544922:
                return -280.924005
            else:
                return -84.766410
    else:
        if miles_per_day <= 106.589287:
            if days_receipts <= 6179.500000:
                return -183.664326
            else:
                return 102.965285
        else:
            if days_receipts <= 6072.410156:
                return 95.867608
            else:
                return 344.220256


def tree_6(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 5330.080078:
        if miles <= 413.500000:
            if rec_lt1500 <= 573.214996:
                return -518.448415
            else:
                return -310.064232
        else:
            if days_receipts <= 3535.100098:
                return -259.016530
            else:
                return 4.089056
    else:
        if miles_per_day <= 99.750000:
            if miles_per_day <= 97.616070:
                return 74.936070
            else:
                return -800.905271
        else:
            if days <= 6.500000:
                return 239.846185
            else:
                return 393.091898


def tree_7(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 656.204987:
        if miles <= 618.000000:
            if miles_lt400 <= 173.500000:
                return -575.278872
            else:
                return -406.507468
        else:
            if rec_lt1500 <= 550.410004:
                return -264.195570
            else:
                return -63.124374
    else:
        if miles_per_day <= 103.512501:
            if days_receipts <= 6179.500000:
                return -166.985991
            else:
                return 90.339822
        else:
            if days_receipts <= 5885.965088:
                return 74.133373
            else:
                return 318.271027


def tree_8(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 710.089996:
        if miles <= 618.000000:
            if days_receipts <= 1779.494995:
                return -511.717797
            else:
                return -352.569805
        else:
            if days_receipts <= 3456.325073:
                return -234.443768
            else:
                return -44.960088
    else:
        if miles_gt800 <= 0.500000:
            if receipts <= 1229.429993:
                return -40.583176
            else:
                return 138.943891
        else:
            if days <= 6.500000:
                return 208.360918
            else:
                return 365.511567


def tree_9(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 3536.190063:
        if miles <= 328.000000:
            if miles_lt400 <= 125.720001:
                return -544.243657
            else:
                return -438.225991
        else:
            if miles <= 757.500000:
                return -298.799030
            else:
                return -187.060840
    else:
        if miles_400_800 <= 182.500000:
            if days_receipts <= 6179.500000:
                return -172.011770
            else:
                return 65.036715
        else:
            if days_receipts <= 5601.875000:
                return 30.387405
            else:
                return 261.060776


def tree_10(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 839.369995:
        if miles <= 632.000000:
            if miles_lt400 <= 262.970001:
                return -468.102586
            else:
                return -292.881573
        else:
            if days_receipts <= 3244.949951:
                return -218.792636
            else:
                return 23.109726
    else:
        if miles_gt800 <= 0.500000:
            if miles_400_800 <= 393.500000:
                return 78.998762
            else:
                return -785.377232
        else:
            if days <= 6.500000:
                return 199.890440
            else:
                return 360.044693


def tree_11(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 5486.340088:
        if miles_per_day <= 81.171429:
            if receipts <= 828.309998:
                return -373.509491
            else:
                return -147.934466
        else:
            if days_receipts <= 3972.565063:
                return -193.710787
            else:
                return 19.399569
    else:
        if miles <= 607.500000:
            if rec_lt1500 <= 1110.915039:
                return -77.345195
            else:
                return 84.689006
        else:
            if days <= 5.500000:
                return 165.409215
            else:
                return 265.967282


def tree_12(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 4241.949951:
        if miles <= 267.000000:
            if rec_lt1500 <= 340.335007:
                return -493.154962
            else:
                return -376.982558
        else:
            if miles_400_800 <= 391.500000:
                return -250.845468
            else:
                return -135.018595
    else:
        if miles_per_day <= 103.512501:
            if days_receipts <= 6179.500000:
                return -133.974860
            else:
                return 55.069605
        else:
            if days_receipts <= 6174.879883:
                return 49.975598
            else:
                return 245.036229


def tree_13(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 6179.500000:
        if miles_per_day <= 47.166666:
            if days_receipts <= 5953.905029:
                return -337.749974
            else:
                return -837.212933
        else:
            if days_receipts <= 3493.570068:
                return -203.532348
            else:
                return 9.804338
    else:
        if miles_per_day <= 99.750000:
            if miles <= 793.500000:
                return 104.545721
            else:
                return -762.160215
        else:
            if receipts <= 1207.034973:
                return 461.437445
            else:
                return 208.581524


def tree_14(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 4343.074951:
        if miles <= 593.500000:
            if miles <= 263.934998:
                return -370.687221
            else:
                return -259.389103
        else:
            if days_receipts <= 2941.100098:
                return -187.084211
            else:
                return -48.717985
    else:
        if miles_per_day <= 99.437500:
            if miles <= 793.500000:
                return 18.240805
            else:
                return -724.052204
        else:
            if days_receipts <= 7051.479980:
                return 85.217779
            else:
                return 223.425704


def tree_15(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 5403.619873:
        if miles_per_day <= 81.971432:
            if receipts <= 828.309998:
                return -305.455293
            else:
                return -94.140718
        else:
            if days_receipts <= 3023.425049:
                return -190.701261
            else:
                return -19.068870
    else:
        if miles_per_day <= 103.412498:
            if miles_per_day <= 102.662498:
                return 62.586462
            else:
                return -370.269454
        else:
            if days <= 6.500000:
                return 146.393215
            else:
                return 266.655056


def tree_16(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 867.375000:
        if miles_400_800 <= 228.500000:
            if days_receipts <= 3149.500000:
                return -316.886410
            else:
                return -182.556699
        else:
            if days_receipts <= 3244.949951:
                return -151.064597
            else:
                return 16.495501
    else:
        if miles_400_800 <= 194.500000:
            if rec_lt1500 <= 1435.849976:
                return -41.772254
            else:
                return 73.492850
        else:
            if days <= 6.500000:
                return 130.131644
            else:
                return 245.284043


def tree_17(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 724.209991:
        if miles <= 791.500000:
            if miles_lt400 <= 256.470001:
                return -327.588579
            else:
                return -190.306065
        else:
            if days_receipts <= 3244.949951:
                return -129.158856
            else:
                return 27.426419
    else:
        if miles_gt800 <= 0.500000:
            if miles_400_800 <= 393.500000:
                return 50.754210
            else:
                return -703.243119
        else:
            if days <= 6.500000:
                return 126.386096
            else:
                return 247.069161


def tree_18(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 867.375000:
        if miles_400_800 <= 228.500000:
            if miles_lt400 <= 216.040001:
                return -323.300952
            else:
                return -209.413146
        else:
            if days_receipts <= 4167.275024:
                return -114.729283
            else:
                return 50.329772
    else:
        if miles_per_day <= 103.512501:
            if miles_per_day <= 102.662498:
                return 40.900506
            else:
                return -246.856361
        else:
            if days <= 6.500000:
                return 121.708793
            else:
                return 248.324900


def tree_19(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 867.375000:
        if miles_gt800 <= 33.500000:
            if days_receipts <= 1666.209961:
                return -310.718317
            else:
                return -170.398014
        else:
            if days_receipts <= 954.625000:
                return -150.285715
            else:
                return 24.043659
    else:
        if miles <= 307.500000:
            if days_receipts <= 6270.475098:
                return -171.803918
            else:
                return 17.036356
        else:
            if is_five <= 0.500000:
                return 170.979377
            else:
                return 66.986165


def tree_20(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 5311.600098:
        if miles <= 416.500000:
            if rec_lt1500 <= 833.510010:
                return -272.570341
            else:
                return -83.676407
        else:
            if rec_lt1500 <= 500.494995:
                return -155.561298
            else:
                return -11.818951
    else:
        if miles_400_800 <= 194.500000:
            if rec_lt1500 <= 1430.020020:
                return -50.523662
            else:
                return 70.574563
        else:
            if days <= 5.500000:
                return 95.531955
            else:
                return 177.539366


def tree_21(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 828.309998:
        if miles_lt400 <= 262.970001:
            if receipts <= 709.779999:
                return -280.334575
            else:
                return -512.719173
        else:
            if days_receipts <= 2478.429932:
                return -171.520612
            else:
                return -56.504803
    else:
        if miles <= 596.000000:
            if receipts <= 1430.020020:
                return -48.172334
            else:
                return 62.390159
        else:
            if days <= 5.500000:
                return 79.316370
            else:
                return 177.883782


def tree_22(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 6178.185059:
        if miles_lt400 <= 216.040001:
            if miles_per_day <= 33.166666:
                return -243.220968
            else:
                return -587.509735
        else:
            if receipts <= 566.375000:
                return -160.746948
            else:
                return 0.765366
    else:
        if miles_per_day <= 60.708334:
            if miles <= 474.500000:
                return 45.181761
            else:
                return -832.665529
        else:
            if rec_lt1500 <= 1202.614990:
                return 267.155974
            else:
                return 125.356780


def tree_23(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 656.204987:
        if miles <= 239.000000:
            if days_receipts <= 2231.329956:
                return -323.792409
            else:
                return -225.043383
        else:
            if miles <= 628.500000:
                return -169.466539
            else:
                return -89.851400
    else:
        if miles_per_day <= 60.708334:
            if miles_per_day <= 59.312500:
                return -1.344249
            else:
                return -791.032252
        else:
            if days_receipts <= 6936.744873:
                return 53.617035
            else:
                return 136.282680


def tree_24(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 867.375000:
        if miles_lt400 <= 265.904999:
            if miles_per_day <= 11.999404:
                return -339.500388
            else:
                return -230.495649
        else:
            if receipts <= 500.494995:
                return -141.345245
            else:
                return -30.365948
    else:
        if miles_gt800 <= 0.500000:
            if miles <= 792.000000:
                return 43.217962
            else:
                return -709.528088
        else:
            if days <= 6.500000:
                return 99.265730
            else:
                return 188.399836


def tree_25(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 4247.284912:
        if miles <= 263.934998:
            if miles_per_day <= 40.475000:
                return -211.500867
            else:
                return -328.691203
        else:
            if miles_400_800 <= 391.500000:
                return -134.787475
            else:
                return -54.144372
    else:
        if miles <= 663.500000:
            if days_receipts <= 11432.660156:
                return -43.607336
            else:
                return 75.754297
        else:
            if days <= 6.500000:
                return 80.852911
            else:
                return 158.185492


def tree_26(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 6178.185059:
        if miles_lt400 <= 261.500000:
            if rec_lt1500 <= 1090.774994:
                return -219.532286
            else:
                return -758.900609
        else:
            if rec_lt1500 <= 500.494995:
                return -123.635053
            else:
                return 0.070298
    else:
        if miles_gt800 <= 120.000000:
            if days <= 7.500000:
                return 76.151381
            else:
                return -16.620546
        else:
            if receipts <= 1200.519958:
                return 310.133742
            else:
                return 116.038170


def tree_27(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 615.134979:
        if miles <= 263.934998:
            if days <= 6.500000:
                return -261.141036
            else:
                return -174.536644
        else:
            if days_receipts <= 2207.175049:
                return -144.763566
            else:
                return -73.128714
    else:
        if miles_per_day <= 107.875000:
            if days_receipts <= 9402.100098:
                return -45.367325
            else:
                return 42.574368
        else:
            if days_receipts <= 5601.875000:
                return 2.519445
            else:
                return 118.054693


def tree_28(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 3899.160034:
        if miles <= 628.500000:
            if days_receipts <= 1725.544983:
                return -225.715732
            else:
                return -142.402723
        else:
            if days <= 6.500000:
                return -88.418275
            else:
                return 36.396566
    else:
        if miles_400_800 <= 196.000000:
            if days_receipts <= 6179.500000:
                return -107.967446
            else:
                return 13.804462
        else:
            if receipts <= 1011.200012:
                return 28.120324
            else:
                return 107.043642


def tree_29(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 6178.185059:
        if days_receipts <= 6139.145020:
            if days_receipts <= 3055.700073:
                return -127.502364
            else:
                return -20.254805
        else:
            return -713.288840
    else:
        if miles_per_day <= 99.987499:
            if miles <= 793.500000:
                return 39.053216
            else:
                return -688.610831
        else:
            if rec_lt1500 <= 1115.809998:
                return 258.467894
            else:
                return 96.682421


def tree_30(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 828.309998:
        if miles <= 628.500000:
            if miles_lt400 <= 216.040001:
                return -205.376071
            else:
                return -120.548874
        else:
            if days_receipts <= 4643.765137:
                return -45.639569
            else:
                return 71.124014
    else:
        if miles <= 596.000000:
            if miles_400_800 <= 78.500000:
                return 21.773434
            else:
                return -87.248728
        else:
            if is_five <= 0.500000:
                return 124.244650
            else:
                return 44.228740


def tree_31(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 6178.185059:
        if miles <= 573.500000:
            if days_receipts <= 5953.905029:
                return -130.889651
            else:
                return -678.713070
        else:
            if days_receipts <= 4407.734863:
                return -39.197982
            else:
                return 44.155294
    else:
        if miles_per_day <= 38.437500:
            if rec_lt1500 <= 1004.709961:
                return -122.564427
            else:
                return -4.579935
        else:
            if is_five <= 0.500000:
                return 108.816552
            else:
                return 24.987186


def tree_32(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 3899.160034:
        if miles_lt400 <= 263.934998:
            if receipts <= 703.315002:
                return -186.347838
            else:
                return -387.771701
        else:
            if miles <= 791.500000:
                return -110.913245
            else:
                return -16.142550
    else:
        if miles_per_day <= 44.428572:
            if miles_per_day <= 39.073000:
                return -21.326665
            else:
                return -397.604437
        else:
            if miles <= 993.500000:
                return 46.768563
            else:
                return 117.740919


def tree_33(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 6178.185059:
        if miles_per_day <= 54.600000:
            if receipts <= 1166.979980:
                return -134.498879
            else:
                return -624.897195
        else:
            if rec_lt1500 <= 500.494995:
                return -104.525383
            else:
                return 13.647013
    else:
        if miles_per_day <= 60.708334:
            if miles <= 474.500000:
                return 26.709792
            else:
                return -758.818565
        else:
            if days <= 5.500000:
                return 45.200502
            else:
                return 110.361793


def tree_34(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 939.264984:
        if miles_per_day <= 56.600000:
            if rec_lt1500 <= 139.864998:
                return -245.647821
            else:
                return -129.754721
        else:
            if miles_gt800 <= 33.500000:
                return -62.137437
            else:
                return 2.487877
    else:
        if miles_gt800 <= 78.500000:
            if days <= 7.500000:
                return 56.546274
            else:
                return -36.098667
        else:
            if receipts <= 1207.034973:
                return 257.126966
            else:
                return 77.718387


def tree_35(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 4471.620117:
        if miles <= 628.500000:
            if days_receipts <= 1825.729980:
                return -164.814577
            else:
                return -90.646216
        else:
            if days <= 5.500000:
                return -80.038411
            else:
                return -14.604844
    else:
        if miles_per_day <= 40.073000:
            if miles_per_day <= 38.510500:
                return -26.060110
            else:
                return -596.479649
        else:
            if miles <= 993.500000:
                return 45.700208
            else:
                return 112.439784


def tree_36(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 828.309998:
        if miles <= 843.000000:
            if miles <= 265.904999:
                return -149.808905
            else:
                return -74.816432
        else:
            if rec_lt1500 <= 648.989990:
                return -18.121778
            else:
                return 70.593222
    else:
        if miles_per_day <= 107.875000:
            if miles_per_day <= 102.662498:
                return 14.230246
            else:
                return -196.719549
        else:
            if days <= 6.500000:
                return 44.181976
            else:
                return 109.333484


def tree_37(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 6178.185059:
        if receipts <= 1201.140015:
            if days_receipts <= 2781.849976:
                return -101.302688
            else:
                return -5.833612
        else:
            return -567.367178
    else:
        if miles <= 890.000000:
            if days <= 7.500000:
                return 57.591757
            else:
                return -6.676103
        else:
            if rec_gt1500_sq <= 160925.789062:
                return 150.116204
            else:
                return 29.348497


def tree_38(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1010.255005:
        if miles_400_800 <= 357.500000:
            if days_receipts <= 1600.409973:
                return -158.788737
            else:
                return -64.899086
        else:
            if miles <= 1101.000000:
                return -6.607875
            else:
                return 63.270903
    else:
        if rec_lt1500 <= 1045.975037:
            if miles_gt800 <= 112.500000:
                return 42.521057
            else:
                return 271.841013
        else:
            if days <= 7.500000:
                return 63.385076
            else:
                return 2.205124


def tree_39(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 914.390015:
        if miles <= 757.500000:
            if days_receipts <= 1680.279968:
                return -133.915278
            else:
                return -65.153145
        else:
            if miles_gt800 <= 308.000000:
                return -11.091404
            else:
                return 70.868605
    else:
        if days <= 7.500000:
            if days <= 5.500000:
                return 13.093014
            else:
                return 102.767394
        else:
            if miles_per_day <= 100.750000:
                return -64.564717
            else:
                return 62.595609


def tree_40(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 828.309998:
        if miles <= 265.904999:
            if receipts <= 769.829987:
                return -124.058498
            else:
                return -381.317559
        else:
            if days_receipts <= 1879.679993:
                return -90.941701
            else:
                return -24.201802
    else:
        if miles_per_day <= 107.875000:
            if miles_per_day <= 99.187500:
                return 13.921004
            else:
                return -124.421609
        else:
            if rec_gt1500_sq <= 107000.300781:
                return 87.928377
            else:
                return 14.162960


def tree_41(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 914.390015:
        if miles_400_800 <= 225.000000:
            if days_receipts <= 979.389984:
                return -176.604449
            else:
                return -79.698440
        else:
            if days <= 6.500000:
                return -28.503508
            else:
                return 29.210455
    else:
        if miles <= 890.000000:
            if days <= 7.500000:
                return 41.159653
            else:
                return -42.719861
        else:
            if receipts <= 1206.039978:
                return 220.346683
            else:
                return 50.369115


def tree_42(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1024.000000:
        if miles_per_day <= 54.312500:
            if miles_lt400 <= 14.500000:
                return -270.840229
            else:
                return -97.769654
        else:
            if days_receipts <= 3478.320068:
                return -51.876433
            else:
                return 7.403837
    else:
        if days <= 5.500000:
            if miles <= 518.500000:
                return -71.331971
            else:
                return 22.704773
        else:
            if receipts <= 1061.139954:
                return 230.898730
            else:
                return 50.541848


def tree_43(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 4241.949951:
        if miles_400_800 <= 228.500000:
            if miles_lt400 <= 14.500000:
                return -257.298217
            else:
                return -79.137226
        else:
            if days <= 5.500000:
                return -41.952206
            else:
                return 19.392159
    else:
        if miles <= 569.500000:
            if miles_400_800 <= 78.500000:
                return 17.948536
            else:
                return -166.576202
        else:
            if miles <= 1183.500000:
                return 51.486287
            else:
                return 166.846106


def tree_44(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 885.850006:
        if miles <= 265.904999:
            if miles_per_day <= 41.666666:
                return -90.109816
            else:
                return -191.825314
        else:
            if miles_gt800 <= 301.000000:
                return -34.718658
            else:
                return 89.897737
    else:
        if miles_lt400 <= 202.000000:
            if miles_per_day <= 35.739666:
                return -7.186528
            else:
                return -542.907585
        else:
            if receipts <= 1615.270020:
                return 67.781762
            else:
                return 25.419809


def tree_45(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1011.200012:
        if miles <= 247.500000:
            if miles_per_day <= 40.475000:
                return -88.261328
            else:
                return -198.281001
        else:
            if receipts <= 663.085022:
                return -47.985221
            else:
                return 14.357441
    else:
        if miles <= 622.000000:
            if miles <= 478.500000:
                return 28.087696
            else:
                return -96.015665
        else:
            if receipts <= 1206.039978:
                return 176.348225
            else:
                return 57.370797


def tree_46(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 828.309998:
        if miles_gt800 <= 33.500000:
            if rec_lt1500 <= 783.065002:
                return -65.469402
            else:
                return -345.357146
        else:
            if miles_per_day <= 187.400002:
                return 28.738273
            else:
                return -79.242450
    else:
        if miles <= 751.500000:
            if days_receipts <= 11413.310059:
                return -18.458162
            else:
                return 53.510901
        else:
            if rec_gt1500 <= 401.154999:
                return 85.253072
            else:
                return 4.121753


def tree_47(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles_400_800 <= 397.500000:
        if miles_400_800 <= 392.000000:
            if rec_lt1500 <= 1416.214966:
                return -45.824412
            else:
                return 39.472736
        else:
            return -676.576190
    else:
        if days <= 6.500000:
            if miles <= 1200.500000:
                return 7.146967
            else:
                return 201.997576
        else:
            if days_receipts <= 13169.205078:
                return 113.127099
            else:
                return -7.547532


def tree_48(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 3459.869995:
        if miles_lt400 <= 182.000000:
            if rec_lt1500 <= 294.285004:
                return -136.135188
            else:
                return -83.288651
        else:
            if is_five <= 0.500000:
                return -5.480830
            else:
                return -69.644472
    else:
        if miles_lt400 <= 221.540001:
            if miles_per_day <= 36.872999:
                return -21.674870
            else:
                return -513.952462
        else:
            if miles <= 1183.500000:
                return 25.805173
            else:
                return 151.945839


def tree_49(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1030.984985:
        if miles <= 833.500000:
            if days_receipts <= 1680.279968:
                return -101.445066
            else:
                return -41.654629
        else:
            if miles_per_day <= 186.700005:
                return 25.931105
            else:
                return -62.125850
    else:
        if miles_per_day <= 40.073000:
            if miles_per_day <= 38.510500:
                return -5.374802
            else:
                return -488.254839
        else:
            if receipts <= 1067.000000:
                return 224.201309
            else:
                return 46.268552


def tree_50(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 663.085022:
        if miles <= 173.500000:
            if rec_lt1500 <= 294.285004:
                return -131.005330
            else:
                return -80.726524
        else:
            if days <= 5.500000:
                return -63.030302
            else:
                return -21.532089
    else:
        if miles <= 310.500000:
            if miles_per_day <= 33.952143:
                return -4.407315
            else:
                return -85.650988
        else:
            if is_five <= 0.500000:
                return 48.231281
            else:
                return 1.885078


def tree_51(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 939.264984:
        if miles_per_day <= 97.833332:
            if days_receipts <= 1836.590027:
                return -84.459983
            else:
                return -42.278533
        else:
            if miles_per_day <= 104.583332:
                return 61.455853
            else:
                return -14.381257
    else:
        if miles_gt800 <= 120.000000:
            if days <= 7.500000:
                return 25.669792
            else:
                return -24.419064
        else:
            if rec_gt1500 <= 310.790009:
                return 108.134003
            else:
                return 14.804135


def tree_52(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles_gt800 <= 0.500000:
        if miles_400_800 <= 394.500000:
            if days_receipts <= 11413.310059:
                return -29.113712
            else:
                return 43.157074
        else:
            return -647.541677
    else:
        if days <= 6.500000:
            if receipts <= 467.915009:
                return -66.543918
            else:
                return 17.276897
        else:
            if days_receipts <= 17896.235352:
                return 72.681871
            else:
                return -64.503015


def tree_53(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1024.000000:
        if miles_per_day <= 54.600000:
            if days_receipts <= 4907.169922:
                return -49.491157
            else:
                return -129.050930
        else:
            if miles_per_day <= 207.800003:
                return -15.903477
            else:
                return -122.586039
    else:
        if rec_lt1500 <= 1038.619995:
            if days_receipts <= 7717.194824:
                return 191.667909
            else:
                return 265.552988
        else:
            if is_five <= 0.500000:
                return 39.129445
            else:
                return -4.597833


def tree_54(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 4165.474976:
        if miles_gt800 <= 197.000000:
            if days_receipts <= 1900.709961:
                return -81.496455
            else:
                return -28.806933
        else:
            if days_receipts <= 3679.895020:
                return 61.224096
            else:
                return 12.509616
    else:
        if miles <= 310.500000:
            if days_receipts <= 9005.665039:
                return -47.122894
            else:
                return 3.465493
        else:
            if rec_gt1500 <= 353.440002:
                return 51.231485
            else:
                return 9.650649


def tree_55(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 885.850006:
        if days_receipts <= 1206.974976:
            if is_five <= 0.500000:
                return -52.896322
            else:
                return -139.632936
        else:
            if miles_per_day <= 54.479166:
                return -63.827398
            else:
                return -9.128554
    else:
        if miles <= 957.500000:
            if days <= 7.500000:
                return 17.776796
            else:
                return -27.175609
        else:
            if receipts <= 1397.859985:
                return 148.815712
            else:
                return 16.024087


def tree_56(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 663.085022:
        if is_five <= 0.500000:
            if days_receipts <= 4606.070068:
                return -7.691078
            else:
                return -92.865382
        else:
            if receipts <= 383.470001:
                return -111.576885
            else:
                return -15.071280
    else:
        if miles_lt400 <= 221.540001:
            if miles_per_day <= 33.166666:
                return -6.468479
            else:
                return -380.234763
        else:
            if days <= 7.500000:
                return 39.184708
            else:
                return -10.290079


def tree_57(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 6179.500000:
        if days_receipts <= 6139.145020:
            if miles <= 833.500000:
                return -34.891960
            else:
                return 30.020090
        else:
            return -438.678417
    else:
        if days <= 7.500000:
            if days_receipts <= 10567.594727:
                return 9.431594
            else:
                return 72.424427
        else:
            if days_receipts <= 11257.879883:
                return 42.125228
            else:
                return -44.001202


def tree_58(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles_400_800 <= 168.500000:
        if miles_400_800 <= 79.500000:
            if rec_lt1500 <= 828.309998:
                return -59.263724
            else:
                return 17.204495
        else:
            if miles_400_800 <= 116.500000:
                return -479.644410
            else:
                return -24.339205
    else:
        if days <= 6.500000:
            if receipts <= 467.915009:
                return -49.798669
            else:
                return 8.017562
        else:
            if days <= 7.500000:
                return 68.912784
            else:
                return 2.368691


def tree_59(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1010.255005:
        if miles_lt400 <= 216.040001:
            if miles_per_day <= 33.166666:
                return -47.974460
            else:
                return -252.425863
        else:
            if days_receipts <= 1879.679993:
                return -44.177304
            else:
                return -4.576406
    else:
        if receipts <= 1863.025024:
            if days_receipts <= 12910.944824:
                return 49.703602
            else:
                return -96.299106
        else:
            if days_receipts <= 9402.100098:
                return -782.617576
            else:
                return 8.656650


def tree_60(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles_gt800 <= 90.000000:
        if days_receipts <= 11609.620117:
            if days_receipts <= 11257.879883:
                return -22.040119
            else:
                return -351.369688
        else:
            if miles_400_800 <= 377.000000:
                return 41.688573
            else:
                return -88.175630
    else:
        if miles_per_day <= 207.800003:
            if rec_lt1500 <= 927.070007:
                return 8.203787
            else:
                return 71.021525
        else:
            if miles <= 1123.000000:
                return -77.080934
            else:
                return 26.954888


def tree_61(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 2478.429932:
        if days <= 5.500000:
            if rec_lt1500 <= 487.860001:
                return -64.129676
            else:
                return -231.683654
        else:
            if miles <= 104.000000:
                return -127.944901
            else:
                return 6.424719
    else:
        if miles_gt800 <= 145.000000:
            if days <= 7.500000:
                return 11.849309
            else:
                return -23.520366
        else:
            if rec_gt1500_sq <= 61610.126953:
                return 66.433448
            else:
                return -9.491691


def tree_62(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles <= 1005.000000:
        if miles <= 247.500000:
            if miles_per_day <= 33.952143:
                return -30.377776
            else:
                return -146.417651
        else:
            if miles_lt400 <= 389.500000:
                return 30.777046
            else:
                return -7.905667
    else:
        if miles_per_day <= 210.400002:
            if days_receipts <= 18275.315430:
                return 74.252669
            else:
                return -95.318363
        else:
            if receipts <= 427.615013:
                return -136.148621
            else:
                return -33.088472


def tree_63(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 1011.200012:
        if miles_lt400 <= 121.000000:
            if rec_lt1500 <= 298.690002:
                return -130.714009
            else:
                return -40.090608
        else:
            if receipts <= 43.830000:
                return -177.469191
            else:
                return -15.092600
    else:
        if receipts <= 1212.919983:
            if miles_gt800 <= 23.000000:
                return 10.759666
            else:
                return 172.184179
        else:
            if days_receipts <= 6179.500000:
                return -172.931916
            else:
                return 21.277672


def tree_64(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 2207.175049:
        if is_five <= 0.500000:
            if miles <= 104.000000:
                return -113.493067
            else:
                return 4.189280
        else:
            if receipts <= 286.225006:
                return -116.857370
            else:
                return -54.860383
    else:
        if miles <= 1174.500000:
            if miles_per_day <= 34.455000:
                return -17.733209
            else:
                return 13.708934
        else:
            if miles_per_day <= 168.642860:
                return 70.664907
            else:
                return 168.095090


def tree_65(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 2207.175049:
        if days <= 5.500000:
            if days_receipts <= 1027.850006:
                return -119.697292
            else:
                return -37.989220
        else:
            if miles <= 796.000000:
                return -40.537251
            else:
                return 77.810164
    else:
        if days <= 7.500000:
            if days_receipts <= 10214.915039:
                return 6.744359
            else:
                return 48.378001
        else:
            if receipts <= 1407.234985:
                return 19.313971
            else:
                return -59.891912


def tree_66(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 1011.200012:
        if miles_lt400 <= 306.820007:
            if miles_lt400 <= 30.500000:
                return -127.587363
            else:
                return -54.326725
        else:
            if days_receipts <= 6898.870117:
                return -6.592543
            else:
                return -97.234168
    else:
        if days <= 5.500000:
            if rec_gt1500_sq <= 137163.828125:
                return 14.426345
            else:
                return -46.459098
        else:
            if days_receipts <= 16990.819336:
                return 49.525398
            else:
                return -30.499916


def tree_67(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles_lt400 <= 204.500000:
        if miles <= 202.000000:
            if miles_per_day <= 39.393999:
                return -32.133046
            else:
                return 131.488071
        else:
            return -254.587723
    else:
        if receipts <= 706.744995:
            if miles_gt800 <= 204.500000:
                return -19.593775
            else:
                return 31.448984
        else:
            if receipts <= 1853.440002:
                return 29.059995
            else:
                return -2.759837


def tree_68(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles_400_800 <= 397.500000:
        if miles <= 794.500000:
            if days_receipts <= 11413.310059:
                return -20.988703
            else:
                return 34.433588
        else:
            return -607.616696
    else:
        if miles_per_day <= 104.062500:
            if miles <= 809.000000:
                return 75.968344
            else:
                return 167.660246
        else:
            if miles <= 1183.500000:
                return 9.983514
            else:
                return 91.433268


def tree_69(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 1024.000000:
        if days_receipts <= 6683.084961:
            if days_receipts <= 1879.679993:
                return -42.344772
            else:
                return -4.978137
        else:
            if rec_lt1500 <= 1014.614990:
                return -87.882939
            else:
                return -146.275838
    else:
        if miles_per_day <= 40.073000:
            if miles_per_day <= 39.073000:
                return -3.423365
            else:
                return -402.700776
        else:
            if miles_per_day <= 112.962502:
                return 59.929753
            else:
                return 12.133079


def tree_70(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles_400_800 <= 354.500000:
        if days_receipts <= 11413.310059:
            if days_receipts <= 11257.879883:
                return -21.992760
            else:
                return -674.527501
        else:
            if miles <= 299.500000:
                return -4.873631
            else:
                return 65.238429
    else:
        if rec_gt1500_sq <= 409178.218750:
            if miles_per_day <= 115.714287:
                return 98.942756
            else:
                return 18.990437
        else:
            if days <= 7.500000:
                return -6.759707
            else:
                return -68.550844


def tree_71(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if is_five <= 0.500000:
        if receipts <= 1024.000000:
            if days_receipts <= 4777.360107:
                return 9.689901
            else:
                return -41.951178
        else:
            if days_receipts <= 13036.489746:
                return 45.444544
            else:
                return -5.237290
    else:
        if receipts <= 383.470001:
            if miles_400_800 <= 106.500000:
                return -68.960227
            else:
                return -118.419887
        else:
            if miles <= 221.500000:
                return -73.538480
            else:
                return -4.353389


def tree_72(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles <= 993.000000:
        if receipts <= 1880.419983:
            if receipts <= 1867.594971:
                return -10.876791
            else:
                return -742.837220
        else:
            if days_receipts <= 16990.819336:
                return 34.750111
            else:
                return -19.524543
    else:
        if miles <= 1038.000000:
            if receipts <= 1200.519958:
                return 185.833475
            else:
                return 33.554583
        else:
            if miles_gt800 <= 383.500000:
                return 2.898930
            else:
                return 76.643902


def tree_73(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles_lt400 <= 221.540001:
        if miles_per_day <= 36.872999:
            if days_receipts <= 1412.890015:
                return -86.250583
            else:
                return -17.103053
        else:
            if miles <= 202.864998:
                return -377.245336
            else:
                return -197.726116
    else:
        if is_five <= 0.500000:
            if miles_gt800 <= 193.500000:
                return 8.791277
            else:
                return 46.818324
        else:
            if receipts <= 205.570000:
                return -108.374184
            else:
                return -9.114150


def tree_74(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 11317.569824:
        if days_receipts <= 11257.879883:
            if receipts <= 1870.265015:
                return 2.451796
            else:
                return -171.521251
        else:
            return -642.969078
    else:
        if receipts <= 1812.449951:
            if days <= 7.500000:
                return 118.480288
            else:
                return 46.058330
        else:
            if days <= 7.500000:
                return 29.931065
            else:
                return -21.486890


def tree_75(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 9927.699707:
        if receipts <= 1867.594971:
            if miles_per_day <= 53.016666:
                return -31.130458
            else:
                return 3.746839
        else:
            if miles_per_day <= 167.599998:
                return -696.663589
            else:
                return -46.320183
    else:
        if days <= 7.500000:
            if receipts <= 1904.214966:
                return 97.564518
            else:
                return 23.868409
        else:
            if receipts <= 1407.234985:
                return 40.222114
            else:
                return -31.316859


def tree_76(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 3101.969971:
        if is_five <= 0.500000:
            if miles <= 668.500000:
                return -31.727463
            else:
                return 37.414024
        else:
            if days_receipts <= 1917.350037:
                return -75.119935
            else:
                return -24.167141
    else:
        if miles <= 221.540001:
            if miles_per_day <= 36.872999:
                return -6.536970
            else:
                return -271.677506
        else:
            if miles <= 476.000000:
                return 33.050467
            else:
                return 5.000301


def tree_77(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 828.309998:
        if receipts <= 816.514984:
            if miles <= 1108.000000:
                return -23.289262
            else:
                return 39.960963
        else:
            return -234.203024
    else:
        if miles_per_day <= 206.400002:
            if miles_per_day <= 152.050003:
                return 2.150475
            else:
                return 31.270196
        else:
            if miles <= 1131.500000:
                return -65.310535
            else:
                return 12.752101


def tree_78(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 1047.940002:
        if miles <= 791.500000:
            if days_receipts <= 6604.235107:
                return -19.452254
            else:
                return -94.847594
        else:
            if receipts <= 912.204987:
                return -6.677089
            else:
                return 69.689445
    else:
        if miles_per_day <= 209.099998:
            if miles <= 199.864998:
                return -19.136698
            else:
                return 28.797029
        else:
            if rec_gt1500 <= 208.369993:
                return -95.039598
            else:
                return -47.368022


def tree_79(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if is_five <= 0.500000:
        if miles <= 946.500000:
            if rec_lt1500 <= 1128.760010:
                return -17.101935
            else:
                return 16.074453
        else:
            if days_receipts <= 12901.875000:
                return 58.338221
            else:
                return -9.671333
    else:
        if rec_lt1500 <= 383.470001:
            if miles <= 467.000000:
                return -46.900492
            else:
                return -102.466815
        else:
            if rec_gt1500 <= 370.264999:
                return 0.560494
            else:
                return -52.979059


def tree_80(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles <= 216.040001:
        if miles_per_day <= 36.872999:
            if miles_per_day <= 11.999404:
                return -49.906858
            else:
                return -1.670094
        else:
            if days_receipts <= 4847.450073:
                return -170.712950
            else:
                return -342.543974
    else:
        if miles <= 510.000000:
            if receipts <= 1378.464966:
                return 2.292730
            else:
                return 64.864894
        else:
            if miles <= 516.500000:
                return -660.978847
            else:
                return 2.247186


def tree_81(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles_per_day <= 11.999404:
        if days_receipts <= 2298.919922:
            if receipts <= 108.084999:
                return -122.710818
            else:
                return -85.153893
        else:
            if days_receipts <= 17942.344727:
                return -37.162518
            else:
                return 9.834121
    else:
        if miles_per_day <= 207.800003:
            if receipts <= 1419.449951:
                return -4.489661
            else:
                return 19.079203
        else:
            if miles_gt800 <= 323.000000:
                return -63.432570
            else:
                return 9.215239


def tree_82(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1030.984985:
        if days_receipts <= 7121.729980:
            if miles_lt400 <= 119.500000:
                return -49.016100
            else:
                return -0.496780
        else:
            if miles_400_800 <= 234.000000:
                return -119.637023
            else:
                return -74.653457
    else:
        if rec_lt1500 <= 1227.849976:
            if miles <= 823.000000:
                return 12.709434
            else:
                return 118.812233
        else:
            if days_receipts <= 6179.500000:
                return -325.192293
            else:
                return 9.294639


def tree_83(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 656.204987:
        if days_receipts <= 4445.040039:
            if is_five <= 0.500000:
                return 0.616406
            else:
                return -27.942433
        else:
            if miles_per_day <= 32.875000:
                return -8.274269
            else:
                return -132.096917
    else:
        if miles <= 890.000000:
            if miles_400_800 <= 110.000000:
                return 19.605398
            else:
                return -10.463435
        else:
            if miles_per_day <= 144.580360:
                return 65.825795
            else:
                return 5.586034


def tree_84(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1419.449951:
        if days_receipts <= 11257.879883:
            if miles_per_day <= 52.012501:
                return -20.746926
            else:
                return 7.458951
        else:
            return -616.319657
    else:
        if receipts <= 1853.440002:
            if days_receipts <= 14540.159668:
                return 55.950553
            else:
                return -62.696218
        else:
            if days_receipts <= 9402.100098:
                return -147.021778
            else:
                return 4.239500


def tree_85(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles <= 990.500000:
        if receipts <= 2487.010010:
            if receipts <= 137.320000:
                return 52.654312
            else:
                return -7.714073
        else:
            return 135.184972
    else:
        if receipts <= 1200.519958:
            if miles_gt800 <= 226.500000:
                return 235.493843
            else:
                return 33.696949
        else:
            if miles <= 1168.500000:
                return -14.194614
            else:
                return 63.649493


def tree_86(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 7.500000:
        if days <= 5.500000:
            if days_receipts <= 1917.350037:
                return -70.385885
            else:
                return -3.177620
        else:
            if days_receipts <= 9930.979980:
                return 4.022880
            else:
                return 40.646972
    else:
        if days_receipts <= 11257.879883:
            if days_receipts <= 8155.239746:
                return -23.626967
            else:
                return 45.061112
        else:
            if rec_lt1500 <= 1433.609985:
                return -585.117971
            else:
                return -37.185828


def tree_87(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 939.264984:
        if miles_per_day <= 12.541571:
            if receipts <= 346.779999:
                return -71.471059
            else:
                return -4.453763
        else:
            if days_receipts <= 147.574997:
                return 96.479547
            else:
                return -9.342934
    else:
        if receipts <= 1212.919983:
            if miles_gt800 <= 23.500000:
                return 7.911065
            else:
                return 118.072460
        else:
            if days_receipts <= 6179.500000:
                return -120.391823
            else:
                return 6.717352


def tree_88(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 79.959999:
        if days <= 6.500000:
            if rec_lt1500 <= 59.755001:
                return -126.087484
            else:
                return -107.229873
        else:
            return -51.666615
    else:
        if days <= 7.500000:
            if days_receipts <= 10567.594727:
                return -6.076115
            else:
                return 30.075371
        else:
            if rec_lt1500 <= 1407.234985:
                return 0.828532
            else:
                return -47.769199


def tree_89(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 683.934998:
        if miles_lt400 <= 362.000000:
            if miles_per_day <= 32.687500:
                return -12.212473
            else:
                return -48.150443
        else:
            if days_receipts <= 370.099991:
                return -119.783110
            else:
                return -80.755520
    else:
        if rec_lt1500 <= 145.025002:
            if days_receipts <= 876.679993:
                return 45.156809
            else:
                return 102.145680
        else:
            if miles_per_day <= 60.625000:
                return -12.941275
            else:
                return 4.494531


def tree_90(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1011.200012:
        if days_receipts <= 6854.879883:
            if rec_lt1500 <= 914.390015:
                return -15.587226
            else:
                return 34.144104
        else:
            if days_receipts <= 7618.679932:
                return -46.625392
            else:
                return -83.356261
    else:
        if receipts <= 1212.919983:
            if miles <= 823.500000:
                return 9.924867
            else:
                return 111.391986
        else:
            if days_receipts <= 6179.500000:
                return -301.360556
            else:
                return 7.796174


def tree_91(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles_gt800 <= 38.000000:
        if miles <= 794.500000:
            if rec_lt1500 <= 1416.214966:
                return -13.461587
            else:
                return 18.273399
        else:
            if days_receipts <= 12689.469727:
                return -30.368715
            else:
                return -588.273888
    else:
        if miles_per_day <= 207.800003:
            if days_receipts <= 12901.875000:
                return 35.039056
            else:
                return -14.294137
        else:
            if miles <= 1131.500000:
                return -61.704450
            else:
                return 23.766257


def tree_92(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 7.500000:
        if days_receipts <= 10535.569824:
            if receipts <= 1867.594971:
                return 1.193185
            else:
                return -98.711259
        else:
            if receipts <= 1904.214966:
                return 88.965069
            else:
                return 12.781058
    else:
        if receipts <= 1407.234985:
            if receipts <= 1110.915039:
                return -19.991110
            else:
                return 43.001992
        else:
            if days_receipts <= 11468.879883:
                return -552.879145
            else:
                return -23.980414


def tree_93(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1653.105042:
        if days_receipts <= 12729.399902:
            if miles_per_day <= 60.625000:
                return -24.120066
            else:
                return 3.485217
        else:
            return -557.661173
    else:
        if days_receipts <= 11917.474609:
            if rec_gt1500_sq <= 160925.789062:
                return 65.712935
            else:
                return 19.640480
        else:
            if miles <= 687.000000:
                return 22.544657
            else:
                return -21.362623


def tree_94(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1879.679993:
        if is_five <= 0.500000:
            if receipts <= 64.524998:
                return 68.769828
            else:
                return -11.937636
        else:
            if miles_per_day <= 101.299999:
                return -40.288810
            else:
                return -88.736836
    else:
        if miles_gt800 <= 383.500000:
            if miles_gt800 <= 345.500000:
                return 4.155156
            else:
                return -53.271307
        else:
            if miles <= 1186.000000:
                return 92.269400
            else:
                return 35.790500


def tree_95(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles <= 755.500000:
        if days_receipts <= 11413.310059:
            if days_receipts <= 11257.879883:
                return -9.779546
            else:
                return -524.236943
        else:
            if miles <= 293.500000:
                return -8.617185
            else:
                return 42.929075
    else:
        if receipts <= 2139.185059:
            if days <= 6.500000:
                return 2.113968
            else:
                return 37.889132
        else:
            if miles <= 1167.500000:
                return -36.280354
            else:
                return 26.976204


def tree_96(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1879.679993:
        if is_five <= 0.500000:
            if miles_gt800 <= 50.500000:
                return -15.569034
            else:
                return 52.788733
        else:
            if miles_per_day <= 101.299999:
                return -30.190135
            else:
                return -84.167822
    else:
        if days <= 7.500000:
            if days_receipts <= 10535.569824:
                return 4.115576
            else:
                return 34.721617
        else:
            if miles_400_800 <= 85.000000:
                return -23.483515
            else:
                return 5.279263


def tree_97(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 7.500000:
        if days <= 6.500000:
            if rec_lt1500 <= 479.290009:
                return -29.855798
            else:
                return 4.480872
        else:
            if miles <= 1003.000000:
                return 17.838682
            else:
                return 66.438307
    else:
        if receipts <= 1401.429993:
            if rec_lt1500 <= 1019.404968:
                return -21.074974
            else:
                return 26.261196
        else:
            if days_receipts <= 11468.879883:
                return -496.850920
            else:
                return -23.326077


def tree_98(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 7.500000:
        if rec_lt1500 <= 193.364998:
            if receipts <= 150.504997:
                return -28.203357
            else:
                return -79.119947
        else:
            if days <= 6.500000:
                return -1.172827
            else:
                return 27.628766
    else:
        if miles_per_day <= 100.750000:
            if miles_per_day <= 89.562500:
                return -29.329935
            else:
                return -530.977988
        else:
            if miles_per_day <= 105.687500:
                return 134.766448
            else:
                return 0.053149


def tree_99(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles_per_day <= 12.541571:
        if days_receipts <= 2298.919922:
            if receipts <= 79.959999:
                return -91.743089
            else:
                return -55.020968
        else:
            if rec_gt1500_sq <= 802906.468750:
                return -20.602384
            else:
                return -76.269363
    else:
        if miles_per_day <= 207.800003:
            if miles_gt800 <= 193.500000:
                return -1.094490
            else:
                return 26.117166
        else:
            if miles_per_day <= 224.599998:
                return -59.122650
            else:
                return 11.532874

