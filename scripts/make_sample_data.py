"""Generate the demo files under data/samples/.

These are SYNTHETIC and exist to exercise the loader's Turkish encoding, ';'
delimiter and ',' decimal handling, plus a clean UTF-8 file for the ML demo.
They are not real measurements and must not be used for any claims.
Run: python scripts/make_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "samples"
SEED = 42


def farm_dataset(n: int = 400) -> pd.DataFrame:
    """Synthetic dairy-goat body condition style table with Turkish headers."""
    rng = np.random.default_rng(SEED)
    yas = rng.integers(1, 9, n)
    canli_agirlik = 35 + 3.2 * yas + rng.normal(0, 4, n)
    sut = np.clip(1.2 + 0.15 * canli_agirlik / 10 + rng.normal(0, 0.4, n), 0.2, None)
    irk = rng.choice(["Saanen", "Kilis", "Halep", "Damascus"], n, p=[0.4, 0.25, 0.2, 0.15])
    bolge = rng.choice(["Çukurova", "Şanlıurfa", "İzmir", "Muğla"], n)
    vks = np.clip(np.round(2 + 0.03 * (canli_agirlik - 45) + rng.normal(0, 0.4, n), 1), 1.0, 5.0)
    df = pd.DataFrame(
        {
            "hayvan_id": [f"K{1000 + i}" for i in range(n)],
            "ırk": irk,
            "bölge": bolge,
            "yaş": yas,
            "canlı_ağırlık_kg": canli_agirlik.round(2),
            "günlük_süt_lt": sut.round(2),
            "vücut_kondisyon_skoru": vks,
        }
    )
    # Inject realistic dirt: missing values and a few outliers.
    df.loc[rng.choice(n, 25, replace=False), "günlük_süt_lt"] = np.nan
    df.loc[rng.choice(n, 10, replace=False), "canlı_ağırlık_kg"] = np.nan
    df.loc[rng.choice(n, 4, replace=False), "canlı_ağırlık_kg"] = 190.0
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = farm_dataset()

    # 1) Turkish Excel-style export: CP1254, ';' delimiter, ',' decimals.
    tr = df.copy()
    for c in ("canlı_ağırlık_kg", "günlük_süt_lt", "vücut_kondisyon_skoru"):
        tr[c] = tr[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}".replace(".", ","))
    tr.to_csv(OUT / "keci_surusu_cp1254.csv", sep=";", index=False, encoding="cp1254")

    # 2) Same data, clean UTF-8 with BOM (what Excel writes when you pick "CSV UTF-8").
    df.to_csv(OUT / "goat_herd_utf8.csv", index=False, encoding="utf-8-sig")

    # 3) Excel workbook.
    df.to_excel(OUT / "goat_herd.xlsx", index=False)
    print(f"Wrote 3 files to {OUT}")


if __name__ == "__main__":
    main()
