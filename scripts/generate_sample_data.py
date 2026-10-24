"""Generate data/sample_sales.csv (deterministic, so results are reproducible)."""

import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

random.seed(42)

PRODUCTS = {
    "Technology": [("Laptop", 899.0), ("Smartphone", 599.0), ("Headphones", 129.0), ("Monitor", 249.0)],
    "Furniture": [("Office Chair", 189.0), ("Standing Desk", 429.0), ("Bookshelf", 119.0)],
    "Office Supplies": [("Notebook Pack", 12.5), ("Printer Paper", 24.0), ("Pen Set", 9.0), ("Stapler", 15.0)],
}
REGIONS = ["North", "South", "East", "West"]
SEGMENTS = ["Consumer", "Corporate", "Small Business"]

rows = []
start = date(2024, 1, 1)
for i in range(1, 401):
    category = random.choice(list(PRODUCTS))
    product, price = random.choice(PRODUCTS[category])
    quantity = random.randint(1, 10)
    discount = random.choice([0.0, 0.0, 0.05, 0.10, 0.15])
    order_date = start + timedelta(days=random.randint(0, 729))
    rows.append({
        "Order ID": f"ORD-{1000 + i}",
        "Order Date": order_date.isoformat(),
        "Region": random.choice(REGIONS),
        "Product": product,
        "Category": category,
        "Customer Segment": random.choice(SEGMENTS),
        "Quantity": quantity,
        "Unit Price": price,
        "Discount": discount,
        "Sales": round(quantity * price * (1 - discount), 2),
    })

df = pd.DataFrame(rows).sort_values("Order Date").reset_index(drop=True)
out = Path(__file__).resolve().parent.parent / "data" / "sample_sales.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Wrote {len(df)} rows to {out}")
