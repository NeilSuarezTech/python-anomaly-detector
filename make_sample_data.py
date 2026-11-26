"""Generate a demo dataset of industrial sensor readings with injected faults.

The data is synthetic: 500 normal readings from a pump (temperature, pressure,
vibration) plus 15 faulty readings (overheating, pressure drops, bearing wear).
"""
from pathlib import Path

import numpy as np
import pandas as pd


def make_readings(n_normal: int = 500, n_faults: int = 15, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    normal = pd.DataFrame({
        "temperature_c": rng.normal(55, 2.5, n_normal),
        "pressure_bar": rng.normal(3.2, 0.15, n_normal),
        "vibration_mm_s": rng.normal(2.0, 0.3, n_normal),
    })
    faults = pd.DataFrame({
        "temperature_c": np.r_[rng.normal(72, 3, 5), rng.normal(56, 2, 5), rng.normal(63, 2, 5)],
        "pressure_bar": np.r_[rng.normal(3.1, 0.2, 5), rng.normal(2.1, 0.2, 5), rng.normal(3.4, 0.2, 5)],
        "vibration_mm_s": np.r_[rng.normal(2.4, 0.3, 5), rng.normal(2.1, 0.3, 5), rng.normal(5.5, 0.6, 5)],
    })
    df = pd.concat([normal, faults], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "timestamp", pd.date_range("2026-01-05 08:00", periods=len(df), freq="5min"))
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(2)
    return df


if __name__ == "__main__":
    out = Path("sample_data/sensor_readings.csv")
    out.parent.mkdir(exist_ok=True)
    make_readings().to_csv(out, index=False)
    print(f"Saved {out}")
