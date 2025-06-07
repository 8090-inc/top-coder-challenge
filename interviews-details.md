# Interview Details Synopsis

Below is a synopsis of the most relevant insights from `INTERVIEWS.md`. It highlights each employee's observations about the legacy reimbursement system and the recurring themes that may help reverse engineer the algorithm.

## Marcus from Sales
- Experiences highly inconsistent reimbursements for seemingly identical trips.
- Suspects calendar effects (certain times of the month more generous).
- Long trips with intensive schedules sometimes pay better, implying a reward for high effort or mileage per day.
- Higher receipts do not guarantee higher reimbursement; overspending can reduce payouts.
- Believes the system may track user history and adjust reimbursements over time.

## Lisa from Accounting
- Notes a base per diem of about **$100 per day** with unexplained adjustments.
- Five‑day trips almost always get a bonus; other lengths do not.
- Mileage reimbursement is tiered: the full rate applies to roughly the first 100 miles, then drops along a curve.
- Receipts show diminishing returns: mid‑range totals ($600‑$800) do best, while high totals gain little and very low totals are penalized.
- Receipt amounts ending in **.49** or **.99** often yield a slight bonus due to rounding behavior.

## Dave from Marketing
- Observes large differences in reimbursement between similar conferences.
- Avoids submitting very small receipt totals because they reduce overall reimbursement below the base per diem.
- Heard rumors about “magic” combinations of trip length, mileage, and spending but finds the system mostly unpredictable.
- Suspects the algorithm might include random elements or city‑specific adjustments.

## Jennifer from HR
- Handles employee complaints about inconsistency; new employees often receive lower reimbursements until they learn optimal practices.
- Recommends avoiding tiny receipts and timing submissions carefully.
- Finds a sweet spot in trip length around **4‑6 days**, particularly five‑day trips.
- Desires more transparency and consistency in the system.

## Kevin from Procurement
- Maintains extensive spreadsheets and has identified clearer patterns:
  - **Efficiency matters** – reimbursements peak when averaging **180‑220 miles per day**.
  - Optimal daily spending varies with trip length: short trips under **$75/day**, medium trips up to **$120/day**, long trips under **$90/day**.
  - Submission timing affects results: Tuesday is best; never submit on Friday.
  - Weak but consistent correlation with lunar phases – new moon submissions average about 4% higher than full moon.
  - Multiple calculation paths suggest interaction effects between trip length, mileage, and spending.
  - Evidence of an adaptive component that changes with user history.

## Common Themes
- Mileage reimbursement decreases non‑linearly after the first ~100 miles.
- Five‑day trips frequently earn a bonus.
- Very small receipt totals are penalized; very large totals have diminishing returns.
- High mileage efficiency (around 180‑220 miles/day) is rewarded.
- Submission timing (day of week and other cycles) influences payouts.
- The system may be adaptive and includes odd rounding behavior on specific cent amounts.

These insights together provide a roadmap for reverse‑engineering the legacy algorithm and explain many of the patterns employees have observed.
