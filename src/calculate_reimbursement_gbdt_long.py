#!/usr/bin/env python3
"""Auto-generated GBDT scorer."""
from typing import Union
def calculate_reimbursement_gbdt_long(trip_duration_days: Union[int,float],
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
        result += 1579.92707447
        return round(result, 2)

def tree_0(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 822.690002:
        if miles <= 602.500000:
            if days_receipts <= 6952.600098:
                return -704.564789
            else:
                return -390.084574
        else:
            if days_receipts <= 6201.945068:
                return -344.204262
            else:
                return -16.832789
    else:
        if miles <= 462.500000:
            if days_receipts <= 11564.875000:
                return -198.737663
            else:
                return 62.315479
        else:
            if miles <= 934.500000:
                return 215.950681
            else:
                return 371.520810


def tree_1(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 9030.705078:
        if miles_400_800 <= 85.000000:
            if rec_lt1500 <= 636.985016:
                return -658.858835
            else:
                return -386.614383
        else:
            if days_receipts <= 6420.389893:
                return -351.976657
            else:
                return -88.952000
    else:
        if miles <= 503.000000:
            if rec_lt1500 <= 1061.010010:
                return -214.350980
            else:
                return 75.979391
        else:
            if miles <= 995.000000:
                return 224.942269
            else:
                return 373.767815


def tree_2(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 822.690002:
        if miles <= 479.500000:
            if rec_lt1500 <= 595.270020:
                return -626.032816
            else:
                return -382.374445
        else:
            if miles_gt800 <= 88.500000:
                return -373.226756
            else:
                return -166.320373
    else:
        if miles_400_800 <= 103.000000:
            if days_receipts <= 13592.429688:
                return -125.141676
            else:
                return 99.521066
        else:
            if miles <= 852.000000:
                return 190.676157
            else:
                return 311.608310


def tree_3(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 822.690002:
        if miles <= 605.000000:
            if days_receipts <= 7123.815186:
                return -598.741014
            else:
                return -287.579773
        else:
            if days_receipts <= 6201.945068:
                return -304.327684
            else:
                return 0.838451
    else:
        if miles_400_800 <= 103.000000:
            if days_receipts <= 13199.189941:
                return -169.820118
            else:
                return 87.900032
        else:
            if miles <= 811.000000:
                return 161.483977
            else:
                return 289.338623


def tree_4(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 822.690002:
        if miles <= 845.000000:
            if days_receipts <= 7341.399902:
                return -526.702259
            else:
                return -273.963842
        else:
            if receipts <= 454.250000:
                return -220.896294
            else:
                return -36.909162
    else:
        if miles <= 497.500000:
            if receipts <= 987.404999:
                return -253.224831
            else:
                return 47.671644
        else:
            if miles <= 900.000000:
                return 161.433111
            else:
                return 298.262760


def tree_5(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 822.690002:
        if miles <= 485.000000:
            if rec_lt1500 <= 581.829987:
                return -556.951701
            else:
                return -329.413456
        else:
            if days_receipts <= 6337.209961:
                return -271.981043
            else:
                return -53.793820
    else:
        if miles_400_800 <= 103.000000:
            if receipts <= 987.404999:
                return -246.099044
            else:
                return 49.327427
        else:
            if miles <= 1101.500000:
                return 198.131547
            else:
                return 404.423010


def tree_6(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 822.690002:
        if miles <= 485.000000:
            if days_receipts <= 7569.209961:
                return -515.148601
            else:
                return -275.046909
        else:
            if days_receipts <= 6090.899902:
                return -282.195617
            else:
                return -49.400163
    else:
        if miles_400_800 <= 103.000000:
            if days_receipts <= 13199.189941:
                return -152.525357
            else:
                return 70.610347
        else:
            if miles_gt800 <= 201.500000:
                return 166.547807
            else:
                return 295.072514


def tree_7(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 822.690002:
        if miles_gt800 <= 78.500000:
            if days_receipts <= 6834.870117:
                return -434.613642
            else:
                return -208.157270
        else:
            if days_receipts <= 5076.044922:
                return -175.961268
            else:
                return -7.431616
    else:
        if miles_400_800 <= 248.500000:
            if days_receipts <= 11571.504883:
                return -137.327319
            else:
                return 75.220130
        else:
            if days <= 12.500000:
                return 177.740076
            else:
                return 301.203503


def tree_8(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 822.690002:
        if miles_400_800 <= 49.500000:
            if days_receipts <= 7569.209961:
                return -485.259785
            else:
                return -217.476927
        else:
            if days_receipts <= 6337.209961:
                return -263.842378
            else:
                return -59.202045
    else:
        if miles <= 806.000000:
            if days_receipts <= 13199.189941:
                return -102.158023
            else:
                return 115.359967
        else:
            if rec_gt1500_sq <= 109350.179688:
                return 293.321491
            else:
                return 157.696511


def tree_9(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 842.734985:
        if miles <= 613.500000:
            if receipts <= 567.014984:
                return -462.956383
            else:
                return -247.929589
        else:
            if days_receipts <= 6201.945068:
                return -184.185035
            else:
                return 19.050822
    else:
        if miles <= 503.000000:
            if days_receipts <= 13592.429688:
                return -105.024439
            else:
                return 64.893723
        else:
            if days <= 12.500000:
                return 140.774099
            else:
                return 285.363391


def tree_10(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 8085.375000:
        if miles_400_800 <= 139.500000:
            if miles_lt400 <= 90.500000:
                return -551.634472
            else:
                return -382.454505
        else:
            if miles <= 955.000000:
                return -234.753792
            else:
                return -66.456074
    else:
        if miles <= 772.500000:
            if days_receipts <= 13199.189941:
                return -102.014927
            else:
                return 91.149451
        else:
            if receipts <= 1824.799988:
                return 260.948861
            else:
                return 147.354381


def tree_11(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 822.690002:
        if miles_gt800 <= 88.500000:
            if rec_lt1500 <= 519.250000:
                return -358.744022
            else:
                return -197.103799
        else:
            if days_receipts <= 4860.494873:
                return -144.288170
            else:
                return -7.046904
    else:
        if miles_400_800 <= 185.000000:
            if rec_lt1500 <= 987.404999:
                return -198.073572
            else:
                return 41.744844
        else:
            if days <= 12.500000:
                return 125.445575
            else:
                return 251.139222


def tree_12(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 8605.635254:
        if miles_400_800 <= 85.000000:
            if miles_lt400 <= 99.424999:
                return -473.135577
            else:
                return -333.453205
        else:
            if miles_gt800 <= 234.500000:
                return -233.279207
            else:
                return -62.092143
    else:
        if miles <= 528.500000:
            if rec_lt1500 <= 1061.010010:
                return -128.557671
            else:
                return 51.761512
        else:
            if days <= 12.500000:
                return 126.515037
            else:
                return 232.763078


def tree_13(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 10527.839844:
        if miles_400_800 <= 79.500000:
            if rec_lt1500 <= 624.440002:
                return -371.296476
            else:
                return -199.898961
        else:
            if days_receipts <= 6337.209961:
                return -189.953046
            else:
                return -1.494008
    else:
        if miles_400_800 <= 301.000000:
            if receipts <= 974.740021:
                return -153.325555
            else:
                return 59.000127
        else:
            if miles <= 995.000000:
                return 141.913559
            else:
                return 229.246142


def tree_14(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 822.690002:
        if miles_lt400 <= 380.000000:
            if receipts <= 679.239990:
                return -378.960280
            else:
                return -167.022158
        else:
            if days <= 11.500000:
                return -213.269190
            else:
                return -84.670500
    else:
        if miles_400_800 <= 103.000000:
            if days_receipts <= 13592.429688:
                return -90.766719
            else:
                return 47.091455
        else:
            if miles_gt800 <= 192.000000:
                return 115.120119
            else:
                return 225.629848


def tree_15(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 8054.530029:
        if miles_gt800 <= 88.500000:
            if days_receipts <= 7123.815186:
                return -308.457231
            else:
                return -148.744108
        else:
            if days_receipts <= 4860.494873:
                return -116.328723
            else:
                return 33.844635
    else:
        if miles_400_800 <= 81.500000:
            if miles <= 474.000000:
                return 3.446061
            else:
                return -615.261469
        else:
            if miles_gt800 <= 292.500000:
                return 111.127316
            else:
                return 279.571130


def tree_16(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 10670.339844:
        if miles <= 539.500000:
            if days_receipts <= 7569.209961:
                return -305.991793
            else:
                return -148.773673
        else:
            if days_receipts <= 6144.889893:
                return -140.637002
            else:
                return 14.402126
    else:
        if miles_gt800 <= 114.500000:
            if days <= 12.500000:
                return 40.918587
            else:
                return 163.513166
        else:
            if receipts <= 1787.724976:
                return 267.873309
            else:
                return 110.171446


def tree_17(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 892.785004:
        if miles_400_800 <= 50.000000:
            if days_receipts <= 5779.140137:
                return -328.742245
            else:
                return -199.687888
        else:
            if days_receipts <= 6201.945068:
                return -160.606914
            else:
                return -12.885384
    else:
        if miles <= 995.000000:
            if days <= 13.500000:
                return 49.336965
            else:
                return 177.914798
        else:
            if rec_gt1500 <= 287.725006:
                return 303.732603
            else:
                return 117.269946


def tree_18(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 10527.839844:
        if miles <= 539.500000:
            if days_receipts <= 6336.614990:
                return -296.271899
            else:
                return -155.316384
        else:
            if days_receipts <= 6201.945068:
                return -140.836179
            else:
                return 20.107345
    else:
        if miles <= 481.500000:
            if miles <= 479.500000:
                return 13.730888
            else:
                return -601.569794
        else:
            if days <= 12.500000:
                return 87.239670
            else:
                return 166.822135


def tree_19(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 10821.705078:
        if miles_gt800 <= 77.000000:
            if days_receipts <= 7123.815186:
                return -242.159203
            else:
                return -122.753909
        else:
            if days_receipts <= 4860.494873:
                return -89.805016
            else:
                return 66.762798
    else:
        if miles <= 540.500000:
            if rec_lt1500 <= 947.005005:
                return -225.702781
            else:
                return 25.597979
        else:
            if miles_gt800 <= 292.500000:
                return 103.830994
            else:
                return 254.294125


def tree_20(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 951.074982:
        if miles_gt800 <= 77.000000:
            if miles_lt400 <= 90.500000:
                return -335.868561
            else:
                return -192.597888
        else:
            if days_receipts <= 4860.494873:
                return -94.524505
            else:
                return 38.873800
    else:
        if miles <= 700.000000:
            if days <= 12.500000:
                return 6.191785
            else:
                return 138.559801
        else:
            if receipts <= 1668.729980:
                return 202.373937
            else:
                return 96.475240


def tree_21(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 10038.929688:
        if miles_lt400 <= 380.000000:
            if rec_lt1500 <= 747.725006:
                return -270.310874
            else:
                return -154.404804
        else:
            if days_receipts <= 6201.945068:
                return -126.508853
            else:
                return -39.314409
    else:
        if miles <= 825.000000:
            if receipts <= 1204.929993:
                return -58.040530
            else:
                return 67.110202
        else:
            if rec_gt1500 <= 246.364998:
                return 196.191736
            else:
                return 78.396614


def tree_22(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 822.059998:
        if miles_gt800 <= 88.500000:
            if miles <= 204.500000:
                return -281.350995
            else:
                return -176.908855
        else:
            if rec_lt1500 <= 466.839996:
                return -85.264504
            else:
                return 58.109383
    else:
        if miles_lt400 <= 230.500000:
            if days_receipts <= 9909.930176:
                return -145.944440
            else:
                return -10.539609
        else:
            if days <= 11.500000:
                return 40.640053
            else:
                return 126.463490


def tree_23(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 10527.839844:
        if miles <= 602.500000:
            if days_receipts <= 6336.614990:
                return -224.123969
            else:
                return -118.501117
        else:
            if receipts <= 540.054993:
                return -94.683509
            else:
                return 34.324789
    else:
        if days <= 12.500000:
            if miles <= 740.500000:
                return 6.372097
            else:
                return 80.242498
        else:
            if receipts <= 1125.650024:
                return 49.636917
            else:
                return 165.018754


def tree_24(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 865.184998:
        if miles_gt800 <= 223.000000:
            if days <= 10.500000:
                return -221.224788
            else:
                return -102.261196
        else:
            if rec_lt1500 <= 23.969999:
                return -207.499853
            else:
                return 17.638783
    else:
        if miles <= 481.500000:
            if miles <= 479.500000:
                return -4.330306
            else:
                return -556.479265
        else:
            if days <= 11.500000:
                return 56.436129
            else:
                return 135.342381


def tree_25(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 8085.375000:
        if miles_400_800 <= 139.500000:
            if days <= 13.500000:
                return -214.685756
            else:
                return -45.857391
        else:
            if days <= 10.500000:
                return -146.484812
            else:
                return -21.277780
    else:
        if miles <= 585.000000:
            if receipts <= 1182.769958:
                return -68.889523
            else:
                return 28.909169
        else:
            if days <= 12.500000:
                return 62.182572
            else:
                return 140.679534


def tree_26(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 865.184998:
        if miles_400_800 <= 49.500000:
            if days <= 11.500000:
                return -225.961681
            else:
                return -124.411653
        else:
            if days <= 10.500000:
                return -136.070744
            else:
                return -12.177187
    else:
        if miles <= 811.000000:
            if days_receipts <= 13199.189941:
                return -85.828430
            else:
                return 40.437321
        else:
            if rec_gt1500_sq <= 113004.179688:
                return 141.658833
            else:
                return 57.058892


def tree_27(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 10038.929688:
        if miles_400_800 <= 338.000000:
            if days <= 13.500000:
                return -171.343488
            else:
                return -7.399502
        else:
            if rec_lt1500 <= 719.084991:
                return -47.431627
            else:
                return 91.269199
    else:
        if miles_400_800 <= 80.000000:
            if days <= 12.500000:
                return -27.633731
            else:
                return 57.887581
        else:
            if receipts <= 2086.205078:
                return 108.462734
            else:
                return 20.295752


def tree_28(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 10670.339844:
        if miles <= 90.500000:
            if days_receipts <= 5567.520020:
                return -300.656281
            else:
                return -124.748092
        else:
            if miles_400_800 <= 338.000000:
                return -117.928678
            else:
                return -48.862763
    else:
        if miles_gt800 <= 195.500000:
            if rec_lt1500 <= 1177.159973:
                return -70.603549
            else:
                return 53.305530
        else:
            if rec_gt1500_sq <= 96992.390625:
                return 192.361769
            else:
                return 51.256494


def tree_29(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 8085.375000:
        if miles <= 539.500000:
            if miles_lt400 <= 66.000000:
                return -276.368484
            else:
                return -149.447194
        else:
            if days_receipts <= 3006.080078:
                return -124.207102
            else:
                return -19.567580
    else:
        if miles_400_800 <= 185.000000:
            if days_receipts <= 14544.200195:
                return -53.961228
            else:
                return 31.253656
        else:
            if miles_gt800 <= 291.500000:
                return 62.500577
            else:
                return 173.669869


def tree_30(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 1000.554993:
        if miles <= 605.000000:
            if days_receipts <= 6336.614990:
                return -174.104460
            else:
                return -92.937767
        else:
            if days_receipts <= 6337.209961:
                return -69.681430
            else:
                return 26.319625
    else:
        if days <= 12.500000:
            if miles <= 740.500000:
                return -0.080747
            else:
                return 65.474527
        else:
            if rec_gt1500 <= 734.709991:
                return 145.766446
            else:
                return 53.671385


def tree_31(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 8085.375000:
        if miles_400_800 <= 139.500000:
            if miles <= 65.500000:
                return -259.933248
            else:
                return -137.419520
        else:
            if miles_per_day <= 52.682692:
                return 55.890280
            else:
                return -71.891494
    else:
        if miles <= 740.500000:
            if miles_400_800 <= 329.500000:
                return 21.457577
            else:
                return -810.805546
        else:
            if receipts <= 1735.190002:
                return 123.564591
            else:
                return 38.891656


def tree_32(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 1000.554993:
        if miles <= 539.500000:
            if days <= 11.500000:
                return -164.433874
            else:
                return -63.614448
        else:
            if days_receipts <= 6201.945068:
                return -48.350259
            else:
                return 31.313221
    else:
        if days <= 10.500000:
            if miles_gt800 <= 347.000000:
                return -21.023351
            else:
                return 213.137572
        else:
            if miles <= 100.000000:
                return -21.019104
            else:
                return 90.041686


def tree_33(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 6250.040039:
        if miles <= 845.000000:
            if days <= 13.500000:
                return -163.064836
            else:
                return 93.649505
        else:
            if days_receipts <= 4860.494873:
                return -52.084365
            else:
                return 54.436080
    else:
        if miles_400_800 <= 81.500000:
            if miles <= 479.500000:
                return -13.176484
            else:
                return -513.359570
        else:
            if miles <= 1016.500000:
                return 39.225565
            else:
                return 117.402008


def tree_34(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 1000.554993:
        if miles <= 599.500000:
            if days <= 10.500000:
                return -164.230893
            else:
                return -92.457506
        else:
            if receipts <= 552.234985:
                return -63.543676
            else:
                return 27.874188
    else:
        if days <= 12.500000:
            if miles_gt800 <= 301.500000:
                return 12.074164
            else:
                return 189.494156
        else:
            if miles_lt400 <= 257.000000:
                return 31.381241
            else:
                return 137.489301


def tree_35(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 954.210022:
        if miles_400_800 <= 84.000000:
            if miles <= 476.500000:
                return -114.326828
            else:
                return -483.068716
        else:
            if days_receipts <= 6201.945068:
                return -62.665684
            else:
                return 19.859955
    else:
        if days <= 11.500000:
            if miles_400_800 <= 340.500000:
                return -39.398547
            else:
                return 54.255159
        else:
            if receipts <= 2180.359985:
                return 91.114881
            else:
                return 15.394242


def tree_36(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 10670.339844:
        if miles_gt800 <= 77.000000:
            if days <= 13.500000:
                return -99.890138
            else:
                return 44.040523
        else:
            if rec_lt1500 <= 514.254990:
                return -27.452297
            else:
                return 79.984496
    else:
        if days <= 12.500000:
            if miles <= 1147.000000:
                return 13.825903
            else:
                return 179.692038
        else:
            if miles_per_day <= 8.343407:
                return -16.469346
            else:
                return 95.081926


def tree_37(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1177.159973:
        if receipts <= 1171.900024:
            if miles <= 877.000000:
                return -63.684914
            else:
                return 35.638542
        else:
            return -776.053708
    else:
        if receipts <= 1735.190002:
            if miles <= 1016.500000:
                return 54.799671
            else:
                return 244.809204
        else:
            if days <= 12.500000:
                return -16.762993
            else:
                return 61.342002


def tree_38(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 951.074982:
        if miles_gt800 <= 77.000000:
            if days_receipts <= 12520.200195:
                return -89.965496
            else:
                return -460.485131
        else:
            if days_receipts <= 11269.244629:
                return -15.320772
            else:
                return 113.748313
    else:
        if days <= 12.500000:
            if miles_400_800 <= 301.000000:
                return -9.822760
            else:
                return 61.116382
        else:
            if receipts <= 2234.710083:
                return 116.592670
            else:
                return 22.869710


def tree_39(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 6277.219971:
        if miles_gt800 <= 234.500000:
            if days <= 12.500000:
                return -124.025801
            else:
                return -12.011402
        else:
            if rec_lt1500 <= 23.969999:
                return -153.204436
            else:
                return 27.796399
    else:
        if miles_400_800 <= 81.500000:
            if miles_400_800 <= 79.500000:
                return -12.734739
            else:
                return -437.460874
        else:
            if days <= 12.500000:
                return 26.109378
            else:
                return 85.686801


def tree_40(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 10038.929688:
        if miles_400_800 <= 30.500000:
            if miles_lt400 <= 90.500000:
                return -152.324418
            else:
                return -81.949433
        else:
            if miles_gt800 <= 223.000000:
                return -39.903420
            else:
                return 19.863908
    else:
        if miles <= 1101.500000:
            if days <= 11.500000:
                return -7.484633
            else:
                return 43.304712
        else:
            if days_receipts <= 27110.239258:
                return 166.663454
            else:
                return -11.111707


def tree_41(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 11269.244629:
        if miles_lt400 <= 81.000000:
            if days_receipts <= 8623.145020:
                return -185.744798
            else:
                return -62.389431
        else:
            if days <= 11.500000:
                return -53.921101
            else:
                return -4.213397
    else:
        if days <= 12.500000:
            if miles <= 1101.500000:
                return 5.522349
            else:
                return 140.143527
        else:
            if days_receipts <= 13771.814941:
                return 12.347591
            else:
                return 87.156193


def tree_42(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 1203.274963:
        if miles_gt800 <= 40.000000:
            if miles_per_day <= 67.080807:
                return -56.174433
            else:
                return -162.831712
        else:
            if days_receipts <= 11269.244629:
                return 1.503931
            else:
                return 183.862888
    else:
        if days <= 10.500000:
            if receipts <= 1427.340027:
                return 78.794118
            else:
                return -34.645191
        else:
            if receipts <= 2185.209961:
                return 71.047820
            else:
                return 16.477413


def tree_43(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 13199.189941:
        if days_receipts <= 12876.625000:
            if days_receipts <= 6277.219971:
                return -75.064191
            else:
                return 2.055463
        else:
            if miles_400_800 <= 40.500000:
                return -46.307260
            else:
                return -574.467167
    else:
        if days <= 12.500000:
            if miles_400_800 <= 147.000000:
                return -21.920987
            else:
                return 40.814500
        else:
            if rec_gt1500_sq <= 356950.546875:
                return 101.469905
            else:
                return 22.316031


def tree_44(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 8054.530029:
        if miles_lt400 <= 81.000000:
            if rec_lt1500 <= 394.769989:
                return -210.312244
            else:
                return -117.237355
        else:
            if days <= 11.500000:
                return -72.402528
            else:
                return 4.290831
    else:
        if miles <= 481.500000:
            if miles_400_800 <= 79.500000:
                return -11.988650
            else:
                return -386.838366
        else:
            if rec_gt1500_sq <= 599341.218750:
                return 54.048043
            else:
                return -40.850517


def tree_45(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 6277.219971:
        if miles_lt400 <= 90.500000:
            if receipts <= 394.769989:
                return -197.694776
            else:
                return -116.955508
        else:
            if days <= 11.500000:
                return -82.594746
            else:
                return -3.507096
    else:
        if miles_400_800 <= 81.500000:
            if miles <= 479.500000:
                return -8.984334
            else:
                return -367.496447
        else:
            if receipts <= 2251.974976:
                return 39.502802
            else:
                return -26.417957


def tree_46(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 1000.134979:
        if miles_400_800 <= 82.000000:
            if miles <= 469.000000:
                return -72.178081
            else:
                return -349.121625
        else:
            if miles_per_day <= 49.207691:
                return 75.154493
            else:
                return -29.485675
    else:
        if days <= 10.500000:
            if miles_400_800 <= 324.500000:
                return -38.147340
            else:
                return 24.219610
        else:
            if miles_lt400 <= 94.500000:
                return -35.083040
            else:
                return 58.003716


def tree_47(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 514.254990:
        if days <= 10.500000:
            if miles_lt400 <= 58.000000:
                return -222.838508
            else:
                return -98.746447
        else:
            if rec_lt1500 <= 106.930000:
                return 100.982343
            else:
                return -46.307615
    else:
        if miles <= 481.500000:
            if miles_400_800 <= 79.500000:
                return -9.338706
            else:
                return -331.665544
        else:
            if days <= 9.500000:
                return -14.569819
            else:
                return 48.204065


def tree_48(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 865.184998:
        if miles_per_day <= 104.000000:
            if days <= 10.500000:
                return -99.841462
            else:
                return -33.644676
        else:
            if days_receipts <= 289.829994:
                return -126.001872
            else:
                return 68.694674
    else:
        if miles_gt800 <= 217.500000:
            if miles <= 102.500000:
                return -41.245523
            else:
                return 19.832039
        else:
            if rec_gt1500 <= 281.850006:
                return 183.981711
            else:
                return 2.601513


def tree_49(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 1203.274963:
        if receipts <= 1169.849976:
            if miles <= 598.500000:
                return -58.302944
            else:
                return 9.612573
        else:
            if miles_lt400 <= 205.000000:
                return -72.717941
            else:
                return -715.628785
    else:
        if receipts <= 2313.835083:
            if days <= 11.500000:
                return 13.875858
            else:
                return 66.110828
        else:
            if days_receipts <= 33548.199219:
                return -38.421105
            else:
                return 49.907896


def tree_50(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 10.500000:
        if receipts <= 529.035004:
            if miles <= 58.000000:
                return -203.789362
            else:
                return -86.875632
        else:
            if miles <= 724.500000:
                return -48.512175
            else:
                return 23.601461
    else:
        if days_receipts <= 13735.165039:
            if rec_lt1500 <= 1145.184998:
                return -8.001906
            else:
                return -300.936763
        else:
            if receipts <= 2235.415039:
                return 53.981819
            else:
                return -7.492945


def tree_51(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 1177.159973:
        if receipts <= 1171.900024:
            if miles_gt800 <= 77.000000:
                return -39.077232
            else:
                return 28.871324
        else:
            return -664.800508
    else:
        if rec_gt1500_sq <= 24051.309570:
            if miles <= 1016.500000:
                return 52.057718
            else:
                return 206.221446
        else:
            if days <= 10.500000:
                return -37.305759
            else:
                return 33.754856


def tree_52(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 564.934998:
        if receipts <= 106.930000:
            if days <= 10.500000:
                return -73.900620
            else:
                return 120.335901
        else:
            if miles <= 1101.000000:
                return -72.768222
            else:
                return 19.631525
    else:
        if days <= 10.500000:
            if miles_400_800 <= 370.500000:
                return -36.137806
            else:
                return 28.010574
        else:
            if days_receipts <= 14515.560059:
                return -11.848343
            else:
                return 41.210870


def tree_53(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 865.184998:
        if miles <= 55.500000:
            if miles <= 51.500000:
                return -112.502033
            else:
                return -239.572622
        else:
            if miles_per_day <= 91.751747:
                return -37.101794
            else:
                return 36.011116
    else:
        if miles_gt800 <= 301.500000:
            if days <= 12.500000:
                return 1.606090
            else:
                return 51.330087
        else:
            if days_receipts <= 27110.239258:
                return 146.403941
            else:
                return -32.045057


def tree_54(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 6336.614990:
        if miles_lt400 <= 90.500000:
            if days_receipts <= 3857.109985:
                return -165.293633
            else:
                return -87.320484
        else:
            if days <= 12.500000:
                return -49.701711
            else:
                return 43.569151
    else:
        if miles <= 1101.000000:
            if miles <= 227.500000:
                return -20.468784
            else:
                return 16.785339
        else:
            if days_receipts <= 27110.239258:
                return 120.705769
            else:
                return -30.442804


def tree_55(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 6277.219971:
        if days <= 10.500000:
            if miles <= 878.000000:
                return -95.831376
            else:
                return -15.978555
        else:
            if miles <= 81.000000:
                return -111.355900
            else:
                return 9.478704
    else:
        if miles_gt800 <= 134.500000:
            if days <= 11.500000:
                return -15.048643
            else:
                return 22.087907
        else:
            if rec_gt1500_sq <= 30895.847656:
                return 106.973066
            else:
                return -6.187927


def tree_56(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles <= 99.424999:
        if rec_lt1500 <= 394.769989:
            if miles <= 51.500000:
                return -111.420121
            else:
                return -188.582297
        else:
            if miles_lt400 <= 12.500000:
                return -107.166206
            else:
                return -42.536013
    else:
        if days <= 11.500000:
            if miles_gt800 <= 352.000000:
                return -21.660409
            else:
                return 162.719695
        else:
            if rec_gt1500 <= 743.230011:
                return 38.209061
            else:
                return -9.436628


def tree_57(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 969.904999:
        if days_receipts <= 12783.530273:
            if miles_per_day <= 79.731060:
                return -37.692298
            else:
                return 3.966092
        else:
            if receipts <= 947.005005:
                return -316.632967
            else:
                return -44.626782
    else:
        if miles <= 104.500000:
            if days <= 13.500000:
                return -53.678560
            else:
                return 25.708708
        else:
            if rec_gt1500_sq <= 80481.523438:
                return 55.437331
            else:
                return 4.381967


def tree_58(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 514.254990:
        if miles_per_day <= 23.245455:
            if days <= 13.500000:
                return -111.174401
            else:
                return 37.329923
        else:
            if days <= 10.500000:
                return -64.972763
            else:
                return 4.394637
    else:
        if days <= 10.500000:
            if miles <= 722.000000:
                return -37.495967
            else:
                return 16.541907
        else:
            if miles_gt800 <= 217.500000:
                return 16.988497
            else:
                return 78.710009


def tree_59(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1177.159973:
        if rec_lt1500 <= 1171.900024:
            if miles_per_day <= 79.731060:
                return -30.781031
            else:
                return 26.732034
        else:
            return -633.673476
    else:
        if rec_gt1500 <= 142.884995:
            if miles_per_day <= 57.333334:
                return 24.636921
            else:
                return 86.609636
        else:
            if days <= 12.500000:
                return -10.910595
            else:
                return 46.904934


def tree_60(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 8085.375000:
        if miles <= 1042.500000:
            if days <= 12.500000:
                return -62.622697
            else:
                return -5.522354
        else:
            if rec_lt1500 <= 23.969999:
                return -118.525956
            else:
                return 33.780186
    else:
        if miles <= 102.500000:
            if receipts <= 807.825012:
                return 81.953398
            else:
                return -41.264924
        else:
            if days <= 9.500000:
                return -24.953715
            else:
                return 28.472576


def tree_61(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1177.159973:
        if rec_lt1500 <= 1169.849976:
            if miles_per_day <= 76.457264:
                return -31.078648
            else:
                return 23.561195
        else:
            return -603.413431
    else:
        if rec_gt1500 <= 157.190002:
            if miles_gt800 <= 216.500000:
                return 39.041676
            else:
                return 174.717751
        else:
            if days <= 10.500000:
                return -29.139862
            else:
                return 17.371860


def tree_62(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles_lt400 <= 90.500000:
        if receipts <= 447.429993:
            if miles_per_day <= 5.722222:
                return -99.067487
            else:
                return -191.441171
        else:
            if miles <= 12.500000:
                return -108.108634
            else:
                return -28.311807
    else:
        if days <= 10.500000:
            if receipts <= 425.180008:
                return -68.270245
            else:
                return -1.033313
        else:
            if miles_per_day <= 97.000000:
                return 15.764297
            else:
                return 104.615209


def tree_63(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 955.595001:
        if days_receipts <= 12767.410156:
            if days_receipts <= 1429.804993:
                return -83.651704
            else:
                return -15.208250
        else:
            if rec_lt1500 <= 947.005005:
                return -300.769603
            else:
                return -36.673047
    else:
        if rec_gt1500 <= 473.225006:
            if miles_gt800 <= 215.500000:
                return 16.236564
            else:
                return 104.222651
        else:
            if days <= 12.500000:
                return -20.644298
            else:
                return 29.692655


def tree_64(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 10.500000:
        if rec_lt1500 <= 519.250000:
            if miles_lt400 <= 69.500000:
                return -136.958861
            else:
                return -62.605668
        else:
            if miles_gt800 <= 347.000000:
                return -16.121694
            else:
                return 94.869315
    else:
        if miles_lt400 <= 342.500000:
            if days <= 13.500000:
                return -24.159311
            else:
                return 37.260578
        else:
            if receipts <= 2300.430054:
                return 34.207140
            else:
                return -25.838493


def tree_65(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles_lt400 <= 90.500000:
        if receipts <= 403.239990:
            if miles_lt400 <= 51.500000:
                return -83.566336
            else:
                return -153.151067
        else:
            if days <= 11.500000:
                return -53.547520
            else:
                return -20.877560
    else:
        if days <= 12.500000:
            if miles_gt800 <= 352.000000:
                return -6.967998
            else:
                return 72.068599
        else:
            if days_receipts <= 13771.814941:
                return 2.848359
            else:
                return 46.912801


def tree_66(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 514.254990:
        if miles <= 1042.500000:
            if days_receipts <= 1601.555054:
                return -94.656514
            else:
                return -34.903332
        else:
            if miles_per_day <= 116.849998:
                return 40.103630
            else:
                return -69.092299
    else:
        if miles <= 683.500000:
            if days <= 13.500000:
                return -14.992307
            else:
                return 62.028985
        else:
            if rec_gt1500_sq <= 25581.970703:
                return 55.100436
            else:
                return 5.512345


def tree_67(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 514.254990:
        if miles_per_day <= 6.088889:
            if miles_per_day <= 5.722222:
                return -92.086414
            else:
                return -164.858037
        else:
            if rec_lt1500 <= 115.295002:
                return 29.253423
            else:
                return -38.927455
    else:
        if miles_gt800 <= 217.500000:
            if miles <= 12.000000:
                return -95.398344
            else:
                return 0.456264
        else:
            if rec_gt1500_sq <= 96992.390625:
                return 99.224299
            else:
                return -28.715554


def tree_68(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1177.159973:
        if rec_lt1500 <= 1171.900024:
            if miles_per_day <= 74.299240:
                return -22.353523
            else:
                return 15.095006
        else:
            return -578.982594
    else:
        if rec_gt1500 <= 142.884995:
            if miles <= 1016.500000:
                return 35.171340
            else:
                return 142.930595
        else:
            if days <= 13.500000:
                return -15.239967
            else:
                return 53.603805


def tree_69(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 3022.480103:
        if days <= 13.500000:
            if miles_gt800 <= 234.500000:
                return -83.071515
            else:
                return 14.790650
        else:
            return 182.483102
    else:
        if miles_lt400 <= 263.479996:
            if days_receipts <= 8460.334961:
                return -56.079944
            else:
                return -5.788112
        else:
            if rec_gt1500_sq <= 540847.875000:
                return 19.987648
            else:
                return -25.147312


def tree_70(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 12.500000:
        if receipts <= 632.975006:
            if miles_gt800 <= 227.000000:
                return -47.612247
            else:
                return 13.278548
        else:
            if receipts <= 1656.859985:
                return 16.158193
            else:
                return -14.111668
    else:
        if days_receipts <= 13771.814941:
            if days_receipts <= 13038.979980:
                return 15.563240
            else:
                return -161.699156
        else:
            if receipts <= 2234.710083:
                return 57.345945
            else:
                return -8.624987


def tree_71(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 969.904999:
        if days_receipts <= 12783.530273:
            if miles_per_day <= 18.121795:
                return -58.719032
            else:
                return -8.201430
        else:
            return -282.504909
    else:
        if days <= 10.500000:
            if rec_gt1500_sq <= 18091.824219:
                return 19.992352
            else:
                return -29.280658
        else:
            if miles_per_day <= 8.369048:
                return -25.043596
            else:
                return 33.895713


def tree_72(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 514.254990:
        if miles_per_day <= 18.121795:
            if miles <= 189.000000:
                return -61.904692
            else:
                return -132.299268
        else:
            if days <= 10.500000:
                return -46.198584
            else:
                return 6.331634
    else:
        if miles_gt800 <= 299.500000:
            if days <= 12.500000:
                return -3.971042
            else:
                return 24.289178
        else:
            if days_receipts <= 27110.239258:
                return 82.688280
            else:
                return -43.302872


def tree_73(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1000.554993:
        if miles <= 482.000000:
            if miles_400_800 <= 76.500000:
                return -33.845569
            else:
                return -269.594122
        else:
            if miles_per_day <= 49.207691:
                return 78.231811
            else:
                return -8.301273
    else:
        if rec_gt1500 <= 142.884995:
            if miles_per_day <= 71.890110:
                return 13.507074
            else:
                return 88.877454
        else:
            if days_receipts <= 22904.115234:
                return -22.924546
            else:
                return 23.694001


def tree_74(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 11.500000:
        if miles_per_day <= 77.013638:
            if miles_per_day <= 66.525253:
                return -19.515165
            else:
                return -103.318894
        else:
            if days_receipts <= 14313.235352:
                return 37.045014
            else:
                return -21.819094
    else:
        if miles_per_day <= 4.886905:
            if miles_per_day <= 3.958333:
                return -19.027196
            else:
                return -79.234528
        else:
            if receipts <= 2243.229980:
                return 25.696059
            else:
                return -14.501602


def tree_75(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles <= 204.500000:
        if miles_lt400 <= 196.169998:
            if days <= 13.500000:
                return -30.872143
            else:
                return 29.665373
        else:
            return -200.452798
    else:
        if days <= 10.500000:
            if rec_lt1500 <= 160.389999:
                return -74.935918
            else:
                return -6.778890
        else:
            if miles_per_day <= 97.000000:
                return 10.769438
            else:
                return 76.202553


def tree_76(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 10.500000:
        if receipts <= 160.389999:
            if miles <= 503.500000:
                return 27.938813
            else:
                return -88.953486
        else:
            if days_receipts <= 15716.655273:
                return -1.455555
            else:
                return -34.401994
    else:
        if miles_lt400 <= 89.500000:
            if receipts <= 494.770004:
                return -74.582013
            else:
                return -24.031074
        else:
            if receipts <= 972.154999:
                return -3.912606
            else:
                return 24.674371


def tree_77(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1341.119995:
        if days_receipts <= 516.420013:
            if rec_lt1500 <= 39.910000:
                return -80.209148
            else:
                return 40.307213
        else:
            if days_receipts <= 1269.539978:
                return -119.974173
            else:
                return -103.947374
    else:
        if rec_lt1500 <= 106.930000:
            if rec_lt1500 <= 101.945000:
                return 155.810134
            else:
                return 177.731808
        else:
            if rec_lt1500 <= 557.109985:
                return -24.130190
            else:
                return 3.875662


def tree_78(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 10.500000:
        if miles <= 738.000000:
            if miles <= 497.500000:
                return -19.636233
            else:
                return -50.935288
        else:
            if miles_400_800 <= 396.500000:
                return 83.166704
            else:
                return -4.444114
    else:
        if rec_lt1500 <= 106.930000:
            if miles <= 65.500000:
                return -20.819872
            else:
                return 134.803624
        else:
            if miles <= 204.500000:
                return -22.130945
            else:
                return 15.191693


def tree_79(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles <= 204.500000:
        if miles_lt400 <= 196.169998:
            if miles_per_day <= 1.145299:
                return -57.568146
            else:
                return -17.787466
        else:
            return -187.921471
    else:
        if days <= 9.500000:
            if rec_lt1500 <= 1373.489990:
                return -3.952634
            else:
                return -48.102260
        else:
            if receipts <= 2303.020020:
                return 16.474255
            else:
                return -35.972198


def tree_80(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 514.254990:
        if days_receipts <= 394.600006:
            if days_receipts <= 331.534988:
                return -59.402241
            else:
                return 86.566297
        else:
            if days_receipts <= 1456.965027:
                return -103.225451
            else:
                return -21.511671
    else:
        if rec_gt1500 <= 813.834991:
            if miles_400_800 <= 252.000000:
                return -0.380192
            else:
                return 25.832642
        else:
            if days_receipts <= 33436.199219:
                return -42.598089
            else:
                return 23.353787


def tree_81(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 10.500000:
        if miles <= 936.000000:
            if miles_per_day <= 95.388889:
                return -23.668743
            else:
                return -90.510999
        else:
            if miles <= 956.500000:
                return 112.832908
            else:
                return -1.961685
    else:
        if miles_per_day <= 99.314396:
            if days_receipts <= 14650.084961:
                return -10.213898
            else:
                return 16.108293
        else:
            if rec_gt1500_sq <= 4327.290527:
                return 166.829833
            else:
                return 56.400788


def tree_82(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles_per_day <= 98.341667:
        if days <= 11.500000:
            if days_receipts <= 5096.030029:
                return -44.471442
            else:
                return -15.139006
        else:
            if miles <= 770.500000:
                return 12.536197
            else:
                return -17.014887
    else:
        if miles_per_day <= 107.740910:
            if receipts <= 151.084999:
                return 167.237498
            else:
                return 57.719458
        else:
            if miles_gt800 <= 342.000000:
                return -37.083637
            else:
                return 73.120317


def tree_83(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles <= 90.500000:
        if rec_lt1500 <= 1226.114990:
            if miles_lt400 <= 12.000000:
                return -78.297996
            else:
                return -34.988309
        else:
            if days_receipts <= 26989.754883:
                return 5.876732
            else:
                return -56.737403
    else:
        if miles_per_day <= 7.357143:
            return 163.953879
        else:
            if days_receipts <= 3022.480103:
                return -31.149873
            else:
                return 5.824532


def tree_84(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles_lt400 <= 204.500000:
        if rec_lt1500 <= 367.500000:
            if days_receipts <= 2901.369995:
                return -65.877393
            else:
                return -151.922796
        else:
            if days_receipts <= 29859.179688:
                return -17.346350
            else:
                return 72.076958
    else:
        if rec_gt1500_sq <= 599341.218750:
            if rec_lt1500 <= 1000.134979:
                return -3.840993
            else:
                return 21.800561
        else:
            if miles_per_day <= 64.733335:
                return -1.802969
            else:
                return -52.612145


def tree_85(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 10.500000:
        if receipts <= 160.389999:
            if miles_per_day <= 55.944445:
                return 36.023780
            else:
                return -99.577262
        else:
            if miles_gt800 <= 299.500000:
                return -14.032402
            else:
                return 66.664052
    else:
        if miles_per_day <= 99.314396:
            if miles <= 768.500000:
                return 16.126134
            else:
                return -4.278163
        else:
            if rec_lt1500 <= 1285.914978:
                return 153.048561
            else:
                return 60.813629


def tree_86(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 12.500000:
        if miles_gt800 <= 342.000000:
            if miles_per_day <= 124.500000:
                return -5.825236
            else:
                return -141.659162
        else:
            if miles <= 1190.500000:
                return 65.338923
            else:
                return -87.794212
    else:
        if receipts <= 1125.650024:
            if receipts <= 930.234985:
                return 10.322406
            else:
                return -25.853097
        else:
            if receipts <= 1210.340027:
                return 180.977436
            else:
                return 20.731077


def tree_87(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 13.500000:
        if miles_400_800 <= 199.000000:
            if days_receipts <= 8460.334961:
                return -36.616295
            else:
                return -7.619960
        else:
            if miles_per_day <= 52.000000:
                return 85.696512
            else:
                return 2.396323
    else:
        if days_receipts <= 3167.640076:
            if days_receipts <= 1427.230042:
                return 145.928007
            else:
                return 157.727627
        else:
            if rec_lt1500 <= 1160.900024:
                return -18.831622
            else:
                return 40.449939


def tree_88(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles <= 379.500000:
        if days_receipts <= 28947.200195:
            if rec_lt1500 <= 747.725006:
                return -31.908456
            else:
                return -6.449237
        else:
            if receipts <= 2319.585083:
                return 127.985150
            else:
                return -1.516228
    else:
        if miles_per_day <= 35.601191:
            if rec_lt1500 <= 713.949982:
                return 133.213384
            else:
                return 85.084869
        else:
            if receipts <= 2274.170044:
                return 9.918724
            else:
                return -26.358619


def tree_89(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 11.500000:
        if miles_400_800 <= 340.500000:
            if miles_400_800 <= 334.000000:
                return -16.096087
            else:
                return -554.931709
        else:
            if days_receipts <= 14313.235352:
                return 29.047458
            else:
                return -12.967325
    else:
        if miles <= 770.500000:
            if miles <= 257.000000:
                return 2.144599
            else:
                return 38.428149
        else:
            if miles_gt800 <= 44.500000:
                return -69.783233
            else:
                return 4.966846


def tree_90(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 11.500000:
        if miles <= 904.500000:
            if miles_per_day <= 95.388889:
                return -16.126329
            else:
                return -106.263344
        else:
            if days_receipts <= 12489.479980:
                return 40.600985
            else:
                return -15.001567
    else:
        if miles_lt400 <= 18.500000:
            if rec_lt1500 <= 1159.029968:
                return -80.604381
            else:
                return -55.557967
        else:
            if miles <= 769.500000:
                return 19.926366
            else:
                return -2.954951


def tree_91(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days <= 11.500000:
        if miles <= 740.500000:
            if miles_400_800 <= 334.000000:
                return -12.979617
            else:
                return -526.378807
        else:
            if receipts <= 566.744995:
                return -23.581077
            else:
                return 20.644570
    else:
        if miles_lt400 <= 18.500000:
            if rec_lt1500 <= 596.699982:
                return -85.534256
            else:
                return -57.724735
        else:
            if rec_gt1500_sq <= 463080.156250:
                return 21.315100
            else:
                return -11.474184


def tree_92(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 865.184998:
        if receipts <= 856.529999:
            if miles_per_day <= 99.522221:
                return -14.687679
            else:
                return 20.374407
        else:
            return -184.343966
    else:
        if days_receipts <= 12170.299805:
            if miles <= 903.000000:
                return 7.717088
            else:
                return 135.180471
        else:
            if days_receipts <= 12904.994629:
                return -84.226776
            else:
                return 4.756267


def tree_93(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 514.254990:
        if days_receipts <= 6472.479980:
            if miles_per_day <= 6.088889:
                return -58.921396
            else:
                return -14.482575
        else:
            if miles_400_800 <= 200.000000:
                return -102.857934
            else:
                return -121.045005
    else:
        if miles <= 644.500000:
            if days <= 11.500000:
                return -21.246346
            else:
                return 13.137972
        else:
            if days_receipts <= 14296.485352:
                return 54.665711
            else:
                return -0.259914


def tree_94(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 514.254990:
        if miles_per_day <= 56.503967:
            if days <= 13.500000:
                return -26.149326
            else:
                return 51.110856
        else:
            if miles_per_day <= 78.619045:
                return -79.548764
            else:
                return -33.328134
    else:
        if miles_per_day <= 72.678570:
            if miles_per_day <= 64.827381:
                return 4.993556
            else:
                return -48.646093
        else:
            if rec_lt1500 <= 1423.965027:
                return 58.095348
            else:
                return -6.555140


def tree_95(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if miles <= 600.500000:
        if rec_lt1500 <= 106.930000:
            if days_receipts <= 1195.510010:
                return -12.822022
            else:
                return 137.530381
        else:
            if days_receipts <= 2995.520020:
                return -64.350846
            else:
                return -8.527859
    else:
        if miles_per_day <= 52.370880:
            if miles_per_day <= 47.153845:
                return -2.203676
            else:
                return 112.732255
        else:
            if miles_per_day <= 57.986013:
                return -44.466257
            else:
                return 3.676653


def tree_96(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 13259.904785:
        if days_receipts <= 12876.625000:
            if miles_lt400 <= 58.000000:
                return -44.117691
            else:
                return -1.870182
        else:
            if miles <= 407.000000:
                return -11.721885
            else:
                return -381.788278
    else:
        if days_receipts <= 14296.485352:
            if receipts <= 1345.489990:
                return 27.551947
            else:
                return 149.764954
        else:
            if rec_lt1500 <= 1202.424988:
                return 92.651423
            else:
                return 0.202396


def tree_97(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if rec_lt1500 <= 865.184998:
        if receipts <= 861.415009:
            if days <= 11.500000:
                return -25.340043
            else:
                return 3.605681
        else:
            if miles_gt800 <= 241.500000:
                return -132.443887
            else:
                return -180.855144
    else:
        if days_receipts <= 12170.299805:
            if miles <= 903.000000:
                return 17.537760
            else:
                return 111.666352
        else:
            if days_receipts <= 13199.189941:
                return -85.733619
            else:
                return 5.237713


def tree_98(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if receipts <= 1973.225037:
        if rec_lt1500 <= 1007.494995:
            if days_receipts <= 13038.979980:
                return -9.008574
            else:
                return -103.674379
        else:
            if receipts <= 1042.600037:
                return 120.508434
            else:
                return 15.876074
    else:
        if miles_per_day <= 120.294441:
            if days <= 13.500000:
                return -20.180062
            else:
                return 11.804764
        else:
            if rec_gt1500 <= 651.755005:
                return -135.178639
            else:
                return -91.525646


def tree_99(days, miles, receipts, is_five, miles_lt400, miles_400_800, miles_gt800, rec_lt1500, rec_gt1500, days_receipts, miles_per_day, rec_gt1500_sq):
    if days_receipts <= 1341.119995:
        if days <= 11.000000:
            if days_receipts <= 1269.539978:
                return -90.525168
            else:
                return -71.710252
        else:
            if days_receipts <= 354.354996:
                return -17.568163
            else:
                return 14.085455
    else:
        if receipts <= 118.005005:
            if rec_lt1500 <= 101.945000:
                return 120.377562
            else:
                return 141.657470
        else:
            if receipts <= 512.884995:
                return -15.693172
            else:
                return 3.754650

