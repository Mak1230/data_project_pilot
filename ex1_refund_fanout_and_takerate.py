"""
    this code of block defines two list charges and refunds 
    calculates the net volume, fees and take rate per merchant
"""

from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pprint import pprint

# mock data set (in memory list of dictionaries)

CHARGES = [
    {"charge_id": "ch_101", "merchant_id": "m_stripe", "amount_cents": 10000, "fee_cents": 320, "status": "succeeded"},
    {"charge_id": "ch_102", "merchant_id": "m_stripe", "amount_cents": 20000, "fee_cents": 640, "status": "succeeded"},
    {"charge_id": "ch_103", "merchant_id": "m_stripe", "amount_cents": 30000, "fee_cents": 960, "status": "succeeded"},
    {"charge_id": "ch_104", "merchant_id": "m_stripe", "amount_cents": 40000, "fee_cents": 1280, "status": "succeeded"},
    {"charge_id": "ch_105", "merchant_id": "m_stripe", "amount_cents": 50000, "fee_cents": 1600, "status": "succeeded"},
    {"charge_id": "ch_106", "merchant_id": "m_stripe", "amount_cents": 60000, "fee_cents": 1920, "status": "failed"},
    {"charge_id": "ch_107", "merchant_id": "m_stripe", "amount_cents": 70000, "fee_cents": 2240, "status": "succeeded"},
    {"charge_id": "ch_108", "merchant_id": "m_stripe", "amount_cents": 80000, "fee_cents": 2560, "status": "succeeded"},
    {"charge_id": "ch_109", "merchant_id": "m_jpmc", "amount_cents": 90000, "fee_cents": 2880, "status": "succeeded"},
    {"charge_id": "ch_110", "merchant_id": "m_jpmc", "amount_cents": 100000, "fee_cents": 3200, "status": "succeeded"},
]

print(CHARGES[1]["merchant_id"])  # m_stripe

REFUNDS = [
    {"refund_id": "re_101", "charge_id": "ch_101", "refund_amount_cents": 2000, "refund_fee_cents": 60, "status": "succeeded"},
    {"refund_id": "re_101", "charge_id": "ch_101", "refund_amount_cents": 4000, "refund_fee_cents": 120, "status": "succeeded"},
    {"refund_id": "re_101", "charge_id": "ch_101", "refund_amount_cents": 6000, "refund_fee_cents": 180, "status": "failed"},
    {"refund_id": "re_104", "charge_id": "ch_101", "refund_amount_cents": 8000, "refund_fee_cents": 240, "status": "succeeded"},
    {"refund_id": "re_105", "charge_id": "ch_102", "refund_amount_cents": 10000, "refund_fee_cents": 300, "status": "succeeded"},
    {"refund_id": "re_106", "charge_id": "ch_102", "refund_amount_cents": 12000, "refund_fee_cents": 360, "status": "succeeded"},
    {"refund_id": "re_107", "charge_id": "ch_103", "refund_amount_cents": 14000, "refund_fee_cents": 420, "status": "succeeded"},
    {"refund_id": "re_108", "charge_id": "ch_103", "refund_amount_cents": 16000, "refund_fee_cents": 480, "status": "succeeded"},
    {"refund_id": "re_109", "charge_id": "ch_109", "refund_amount_cents": 18000, "refund_fee_cents": 540, "status": "succeeded"},
    {"refund_id": "re_110", "charge_id": "ch_109", "refund_amount_cents": 20000, "refund_fee_cents": 600, "status": "succeeded"},
]


def calculate_merchant_metrics(charges: list[dict], refunds: list[dict]) -> tuple[dict, dict]:
    """
    Returns (merchant_totals, refunds_total_by_charge).
    Pre-aggregates refunds per charge_id to avoid the 1-to-many fan-out bug.
    """
    # pre-aggregate successful refunds per charge_id
    refunds_total_by_charge: dict[str, dict] = defaultdict(lambda: {"amount": 0, "fee": 0})
    for r in refunds:
        if r.get("status") == "succeeded":
            c_id = r["charge_id"]
            refunds_total_by_charge[c_id]["amount"] += r["refund_amount_cents"]
            refunds_total_by_charge[c_id]["fee"] += r["refund_fee_cents"]

    # aggregate successful charges per merchant and join pre-aggregated refunds
    merchant_totals = defaultdict(lambda: {"net_volume_cents": 0, "net_fee_cents": 0})
    for c in charges:
        if c.get("status") != "succeeded":
            continue

        m_id = c["merchant_id"]
        c_id = c["charge_id"]
        ref_info = refunds_total_by_charge[c_id]

        net_vol = c["amount_cents"] - ref_info["amount"]
        net_fee = c["fee_cents"] - ref_info["fee"]

        merchant_totals[m_id]["net_volume_cents"] += net_vol
        merchant_totals[m_id]["net_fee_cents"] += net_fee

    # take_rate must be computed BEFORE return (code after return is unreachable)
    for m_id, totals in merchant_totals.items():
        vol = Decimal(totals["net_volume_cents"])
        fee = Decimal(totals["net_fee_cents"])
        totals["take_rate_pct"] = (
            ((fee / vol) * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            if vol != 0
            else Decimal("0.00")
        )

    return merchant_totals, refunds_total_by_charge


# Unpack BOTH return values after calling the function
merchant_totals, refunds_total_by_charge = calculate_merchant_metrics(CHARGES, REFUNDS)

# Only show charges that actually had successful refunds (skip defaultdict zeros)
refunds_view = {
    c_id: amounts
    for c_id, amounts in refunds_total_by_charge.items()
    if amounts["amount"] or amounts["fee"]
}

print("\n=== Refunds total by charge ===")
pprint(refunds_view, sort_dicts=False)

print("\n=== Merchant metrics ===")
pprint(dict(merchant_totals), sort_dicts=False)

print("\n=== Merchant summary (readable) ===")
print(f"{'merchant':<12} {'net_volume_$':>12} {'net_fee_$':>10} {'take_rate_%':>11}")
print("-" * 48)
for m_id, t in merchant_totals.items():
    print(
        f"{m_id:<12} "
        f"{t['net_volume_cents'] / 100:>12,.2f} "
        f"{t['net_fee_cents'] / 100:>10,.2f} "
        f"{t['take_rate_pct']:>11}"
    )
