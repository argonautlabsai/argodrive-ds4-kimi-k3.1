#!/usr/bin/env python3
"""A 6,000-token benchmark prompt with a JSON answer: synthetic daily weather readings of three cities (seeded, so
the same text comes out every time) and the instruction to return one city's rows as JSON. 67 days give 6,051 tokens
with the DeepSeek V4.1 chat template.

usage: make_weather_prompt.py <days> <out file>"""
import datetime
import random
import sys

days, out = int(sys.argv[1]), sys.argv[2]
random.seed(20261122)
cities = [("Lisbon", 21.0, 7.0), ("Vienna", 14.0, 9.0), ("Oslo", 9.0, 11.0)]
rows = ["date,city,temp_max_c,temp_min_c,rain_mm,wind_kmh,humidity_pct,sunshine_h"]
d0 = datetime.date(2026, 3, 1)
for i in range(days):
    d = d0 + datetime.timedelta(days=i)
    for name, tmax, swing in cities:
        hi = tmax + 0.12 * i + random.uniform(-3.5, 3.5)
        lo = hi - swing + random.uniform(-2.0, 2.0)
        rain = max(0.0, random.gauss(1.5, 4.0)) if random.random() < 0.45 else 0.0
        rows.append(f"{d.isoformat()},{name},{hi:.1f},{lo:.1f},{rain:.1f},{random.uniform(4, 38):.0f},"
                    f"{random.uniform(38, 96):.0f},{max(0.0, random.uniform(0, 12) - rain / 3):.1f}")
text = ("Here are daily weather readings for three cities (CSV):\n\n" + "\n".join(rows) + "\n\n"
        "Return the Vienna rows as a JSON array of objects with the fields date, temp_max_c, rain_mm and sunshine_h, "
        "in date order, with no other text.")
open(out, "w").write(text)
print(len(rows) - 1, "rows,", len(text), "characters")
