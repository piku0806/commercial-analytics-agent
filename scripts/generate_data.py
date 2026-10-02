"""Generate a fictional pharma commercial dataset (prescriptions, prescribers, products) as dbt seeds.

    python scripts/generate_data.py
All product names, prescribers and numbers are invented.
"""
from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

SEEDS = Path(__file__).resolve().parents[1] / "dbt" / "seeds"
rng = random.Random(7)

REGIONS = {"Northeast": ["Boston", "New York", "Philadelphia"], "Southeast": ["Atlanta", "Miami", "Charlotte"],
           "Midwest": ["Chicago", "Detroit", "Minneapolis"], "West": ["Los Angeles", "Seattle", "Denver"]}
SPECIALTIES = ["Cardiology", "Endocrinology", "Pulmonology", "Primary Care"]
PRODUCTS = [  # product_id, brand, therapeutic_area, own_brand, price
    ("P1", "Xelvora", "Cardiovascular", True, 412.0), ("P2", "Tamberix", "Cardiovascular", False, 365.0),
    ("P3", "Quorvane", "Diabetes", True, 520.0), ("P4", "Lindoset", "Diabetes", False, 498.0),
    ("P5", "Respiquel", "Respiratory", True, 289.0), ("P6", "Aeronil", "Respiratory", False, 301.0)]
AREA_FOR_SPECIALTY = {"Cardiology": "Cardiovascular", "Endocrinology": "Diabetes", "Pulmonology": "Respiratory"}
PAYERS = ["Commercial", "Medicare", "Medicaid", "Cash"]


def main() -> None:
    SEEDS.mkdir(parents=True, exist_ok=True)
    prescribers = []
    pid = 1
    for region, territories in REGIONS.items():
        for terr in territories:
            for _ in range(10):
                prescribers.append([f"HCP{pid:04d}", f"Prescriber {pid:04d}", rng.choice(SPECIALTIES), terr, region])
                pid += 1
    rx, rid = [], 1
    start = date(2025, 1, 1)
    for day in range(0, 638):                      # 2025-01-01 .. 2026-09-30
        d = start + timedelta(days=day)
        growth = 1 + day / 638 * 0.35               # own brands grow over time
        for _ in range(rng.randint(18, 30)):
            hcp = rng.choice(prescribers)
            area = AREA_FOR_SPECIALTY.get(hcp[2]) or rng.choice(["Cardiovascular", "Diabetes", "Respiratory"])
            options = [p for p in PRODUCTS if p[2] == area]
            weights = [growth if p[3] else 1.0 for p in options]
            if hcp[4] == "Southeast":
                weights = [w * (1.3 if p[3] else 1.0) for w, p in zip(weights, options)]
            prod = rng.choices(options, weights=weights)[0]
            qty = rng.choice([30, 30, 30, 60, 90])
            payer = rng.choices(PAYERS, weights=[50, 30, 15, 5])[0]
            discount = {"Commercial": 0.15, "Medicare": 0.25, "Medicaid": 0.40, "Cash": 0.0}[payer]
            net = round(prod[4] * qty / 30 * (1 - discount), 2)
            rx.append([f"RX{rid:07d}", d.isoformat(), hcp[0], prod[0], payer, qty, net, int(rng.random() < 0.22)])
            rid += 1
    with (SEEDS / "raw_prescribers.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["prescriber_id", "prescriber_name", "specialty", "territory", "region"], *prescribers])
    with (SEEDS / "raw_products.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["product_id", "brand", "therapeutic_area", "is_own_brand", "list_price"],
                                 *[[p[0], p[1], p[2], str(p[3]).lower(), p[4]] for p in PRODUCTS]])
    with (SEEDS / "raw_prescriptions.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["rx_id", "rx_date", "prescriber_id", "product_id", "payer_type", "quantity",
                                  "net_sales", "is_new_rx"], *rx])
    print(f"prescribers={len(prescribers)} prescriptions={len(rx)}")


if __name__ == "__main__":
    main()
