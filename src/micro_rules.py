from typing import Union

def apply_micro_rules(days: Union[int, float], miles: Union[int, float], receipts: Union[int, float], base_pred: float) -> float:
    """
    Apply refined micro-rule offsets to the base prediction.
    Rules:
      1. If receipts > $2000: subtract 900
      2. If receipts > $1800 and days <= 3: subtract 700
      3. If 5 <= days <= 8 and 400 < miles < 900 and receipts > $1400: subtract 700
      4. If receipts > $1400 and miles < 200: subtract 600
      5. If receipts > $1800 and miles > 900: subtract 800
    """
    adj = 0.0
    if receipts > 2000:
        adj -= 900.0
    if receipts > 1800 and days <= 3:
        adj -= 700.0
    if 5 <= days <= 8 and 400 < miles < 900 and receipts > 1400:
        adj -= 700.0
    if receipts > 1400 and miles < 200:
        adj -= 600.0
    if receipts > 1800 and miles > 900:
        adj -= 800.0
    return base_pred + adj 