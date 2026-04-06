#!/usr/bin/env python3
"""
Earth → Mars Porkchop Plot Generator

Computes and renders Δv contour maps for interplanetary transfer windows.
Supports multiple preset mission windows and configurable grid resolution.

Usage:
    python main.py                              # Earth→Mars 2026
    python main.py --window earth-venus-2026    # Earth→Venus 2026
    python main.py --validate                   # run tests first
    python main.py --step 1                     # 1-day grid resolution
"""

from __future__ import annotations

import argparse
import time
import numpy as np

from ephemeris import calendar_to_jd, MU_SUN
from porkchop import compute_dv_grid, plot_porkchop, plot_departure_arrival_dv


# --- Preset mission windows ---

WINDOWS = {
    "earth-mars-2026": {
        "dep_planet": "earth", "arr_planet": "mars",
        "dep_start": (2026, 7, 1), "dep_end": (2027, 3, 1),
        "arr_start": (2027, 1, 1), "arr_end": (2027, 12, 1),
        "step_days": 3, "dv_max": 25.0,
    },
    "earth-mars-2028": {
        "dep_planet": "earth", "arr_planet": "mars",
        "dep_start": (2028, 8, 1), "dep_end": (2029, 4, 1),
        "arr_start": (2029, 2, 1), "arr_end": (2030, 1, 1),
        "step_days": 3, "dv_max": 25.0,
    },
    "earth-venus-2026": {
        "dep_planet": "earth", "arr_planet": "venus",
        "dep_start": (2026, 1, 1), "dep_end": (2026, 10, 1),
        "arr_start": (2026, 4, 1), "arr_end": (2027, 1, 1),
        "step_days": 3, "dv_max": 20.0,
    },
    "earth-jupiter-2028": {
        "dep_planet": "earth", "arr_planet": "jupiter",
        "dep_start": (2028, 1, 1), "dep_end": (2029, 6, 1),
        "arr_start": (2029, 6, 1), "arr_end": (2032, 1, 1),
        "step_days": 7, "dv_max": 40.0,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Porkchop plot generator — Izzo Lambert solver")
    parser.add_argument("--window", "-w", choices=list(WINDOWS.keys()),
                        default="earth-mars-2026", help="Mission window preset")
    parser.add_argument("--validate", "-v", action="store_true", help="Run validation suite first")
    parser.add_argument("--step", "-s", type=int, default=None, help="Grid step in days")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output filename")
    args = parser.parse_args()

    if args.validate:
        from validate import run_all
        run_all()

    cfg = WINDOWS[args.window]
    step = args.step or cfg["step_days"]
    output = args.output or f"porkchop_{args.window}.png"
    output_comp = output.replace(".png", "_components.png")

    dep_dates = np.arange(calendar_to_jd(*cfg["dep_start"]), calendar_to_jd(*cfg["dep_end"]), step)
    arr_dates = np.arange(calendar_to_jd(*cfg["arr_start"]), calendar_to_jd(*cfg["arr_end"]), step)

    total = len(dep_dates) * len(arr_dates)

    print(f"\n{'='*60}")
    print(f"  PORKCHOP PLOT  —  {cfg['dep_planet'].upper()} → {cfg['arr_planet'].upper()}")
    print(f"{'='*60}")
    print(f"  Departure: {cfg['dep_start']}  →  {cfg['dep_end']}")
    print(f"  Arrival:   {cfg['arr_start']}  →  {cfg['arr_end']}")
    print(f"  Grid: {len(dep_dates)} × {len(arr_dates)} = {total:,} Lambert solves")
    print(f"{'='*60}\n")

    t0 = time.time()
    dv_dep, dv_arr, dv_tot, tof_days = compute_dv_grid(
        cfg["dep_planet"], cfg["arr_planet"], dep_dates, arr_dates,
    )
    elapsed = time.time() - t0

    valid = np.isfinite(dv_tot)
    n_valid = valid.sum()

    print(f"\n  Completed in {elapsed:.1f} s  ({n_valid:,}/{total:,} valid)")
    if n_valid > 0:
        idx = np.unravel_index(np.nanargmin(dv_tot), dv_tot.shape)
        print(f"  Min Δv: {dv_tot[idx]:.3f} km/s  "
              f"(dep={dv_dep[idx]:.3f}, arr={dv_arr[idx]:.3f}, TOF={tof_days[idx]:.0f}d)")

    plot_porkchop(dep_dates, arr_dates, dv_dep, dv_arr, dv_tot, tof_days,
                  dep_planet=cfg["dep_planet"], arr_planet=cfg["arr_planet"],
                  save_path=output, dv_max=cfg["dv_max"])

    plot_departure_arrival_dv(dep_dates, arr_dates, dv_dep, dv_arr, dv_tot, tof_days,
                              dep_planet=cfg["dep_planet"], arr_planet=cfg["arr_planet"],
                              save_path=output_comp, dv_max=cfg["dv_max"])

    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()