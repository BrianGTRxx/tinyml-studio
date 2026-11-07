"""Synthetic appliance recordings used ONLY to take the screenshots of the manual.

Each column is drawn independently from a per device distribution (no electrical formulas are
implemented), so the files look plausible but carry no real measurements. Usage:
    python tools/manual/make_sample_data.py OUT_DIR
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# device: (voltage, current, frequency, active_power, reactive_power, apparent_power, power_factor)
DEVICES = {
    "kettle": (120.0, 15.0, 60.0, 1800.0, 20.0, 1810.0, 0.99),
    "fan": (120.0, 0.45, 60.0, 45.0, 28.0, 54.0, 0.84),
    "lamp": (120.0, 0.12, 60.0, 9.0, 11.0, 14.0, 0.62),
}
FILES = [("kettle", 1), ("kettle", 2), ("fan", 1), ("fan", 2), ("lamp", 1)]
COLS = [
    "voltage",
    "current",
    "frequency",
    "active_power",
    "reactive_power",
    "apparent_power",
    "power_factor",
]
SPREAD = [0.6, 0.04, 0.03, 0.03, 0.12, 0.03, 0.01]  # relative noise per column


def main(out: Path, rows: int = 1500) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for k, (dev, n) in enumerate(FILES):
        base = np.array(DEVICES[dev])
        offset = 1 + rng.normal(0, 0.03, len(COLS)) * (n - 1)  # the second session drifts a little
        data = {}
        for j, c in enumerate(COLS):
            sd = SPREAD[j] if c in ("voltage", "frequency") else abs(base[j]) * SPREAD[j]
            data[c] = base[j] * offset[j] + rng.normal(0, sd, rows)
        df = pd.DataFrame(data).round(3)
        df.insert(0, "sensor", f"sensor_{chr(65 + k)}")
        stamps = pd.date_range("2026-03-01 10:00:00", periods=rows, freq="s")
        df.insert(0, "time", stamps.strftime("%Y-%m-%dT%H:%M:%SZ"))
        df.loc[rng.choice(rows, 12, replace=False), "current"] = np.nan  # missing values
        df.loc[rng.choice(rows, 5, replace=False), "power_factor"] = 9.0  # absurd values
        df.to_csv(out / f"{dev}_{n}.csv", index=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
