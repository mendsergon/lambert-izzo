"""
Porkchop Plot Generator

Computes and renders departure/arrival Δv contour maps for interplanetary
transfers. For each (departure, arrival) date pair on a grid, solves
Lambert's problem and computes departure, arrival, and total Δv.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Tuple, Optional

from lambert import solve as lambert_solve
from ephemeris import state_vector, calendar_to_jd, MU_SUN, DAY_S


# --- Grid computation ---

def compute_dv_grid(
    dep_planet: str,
    arr_planet: str,
    dep_dates: np.ndarray,
    arr_dates: np.ndarray,
    mu: float = MU_SUN,
    prograde: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Δv over a 2-D departure × arrival date grid.

    Returns (dv_dep, dv_arr, dv_tot, tof_days) arrays of shape (n_dep, n_arr).
    """
    n_dep = len(dep_dates)
    n_arr = len(arr_dates)

    dv_dep   = np.full((n_dep, n_arr), np.nan)
    dv_arr   = np.full((n_dep, n_arr), np.nan)
    dv_tot   = np.full((n_dep, n_arr), np.nan)
    tof_days = np.full((n_dep, n_arr), np.nan)

    total = n_dep * n_arr
    done = 0

    for i, jd_dep in enumerate(dep_dates):
        r1, v_planet1 = state_vector(dep_planet, jd_dep)

        for j, jd_arr in enumerate(arr_dates):
            done += 1
            if done % 500 == 0:
                print(f"  [{100.0 * done / total:5.1f}%]  {done}/{total}", flush=True)

            tof = (jd_arr - jd_dep) * DAY_S
            if tof <= 0:
                continue

            r2, v_planet2 = state_vector(arr_planet, jd_arr)

            try:
                sols = lambert_solve(r1, r2, tof, mu, prograde=prograde)
            except Exception:
                continue

            if not sols:
                continue

            v1_transfer, v2_transfer = sols[0]

            dv1 = norm(v1_transfer - v_planet1)
            dv2 = norm(v_planet2 - v2_transfer)

            dv_dep[i, j]   = dv1
            dv_arr[i, j]   = dv2
            dv_tot[i, j]   = dv1 + dv2
            tof_days[i, j] = tof / DAY_S

    return dv_dep, dv_arr, dv_tot, tof_days


# --- Date conversion ---

def _jd_to_datetime(jd: float) -> datetime:
    return datetime(2000, 1, 1, 12, 0, 0) + timedelta(days=jd - 2451545.0)


def _jd_array_to_datetimes(jds: np.ndarray):
    return np.array([_jd_to_datetime(jd) for jd in jds])


# --- Plotting ---

def plot_porkchop(
    dep_dates: np.ndarray,
    arr_dates: np.ndarray,
    dv_dep: np.ndarray,
    dv_arr: np.ndarray,
    dv_tot: np.ndarray,
    tof_days: np.ndarray,
    dep_planet: str = "Earth",
    arr_planet: str = "Mars",
    save_path: str = "porkchop.png",
    dv_max: float = 30.0,
    levels: Optional[np.ndarray] = None,
) -> str:
    """
    Render total-Δv porkchop plot with TOF isolines and minimum-Δv annotation.
    """
    dep_dt = _jd_array_to_datetimes(dep_dates)
    arr_dt = _jd_array_to_datetimes(arr_dates)

    dv_plot = np.copy(dv_tot)
    dv_plot[dv_plot > dv_max] = np.nan

    if levels is None:
        levels = np.array([4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 25, 30])
        levels = levels[levels <= dv_max]

    tof_levels = np.arange(100, 650, 50)

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    DD, AA = np.meshgrid(dep_dt, arr_dt, indexing="ij")

    # Total Δv contours
    cf = ax.contourf(DD, AA, dv_plot, levels=levels, cmap="magma_r", extend="both")
    cs = ax.contour(DD, AA, dv_plot, levels=levels,
                    colors="white", linewidths=0.4, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f", colors="white")

    # TOF contours
    cs_tof = ax.contour(DD, AA, tof_days, levels=tof_levels,
                        colors="cyan", linewidths=0.5, linestyles="dashed", alpha=0.4)
    ax.clabel(cs_tof, inline=True, fontsize=6, fmt="%.0f d", colors="cyan")

    cbar = fig.colorbar(cf, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("Total Δv  [km/s]", fontsize=12, color="white")
    cbar.ax.tick_params(colors="white")

    # Minimum Δv annotation
    valid = np.isfinite(dv_tot)
    if valid.any():
        idx = np.unravel_index(np.nanargmin(dv_tot), dv_tot.shape)
        min_dv = dv_tot[idx]
        min_dep = dep_dt[idx[0]]
        min_arr = arr_dt[idx[1]]

        ax.plot(min_dep, min_arr, "w*", markersize=14, zorder=10)
        label_text = (
            f"Δv = {min_dv:.2f} km/s\n"
            f"  dep Δv = {dv_dep[idx]:.2f}\n"
            f"  arr Δv = {dv_arr[idx]:.2f}\n"
            f"  TOF = {tof_days[idx]:.0f} d\n"
            f"  Dep: {min_dep.strftime('%Y-%m-%d')}\n"
            f"  Arr: {min_arr.strftime('%Y-%m-%d')}"
        )
        ax.annotate(
            label_text, xy=(min_dep, min_arr), xytext=(30, 30),
            textcoords="offset points", fontsize=9, color="white",
            bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a2e", ec="white", alpha=0.85),
            arrowprops=dict(arrowstyle="->", color="white", lw=1.2), zorder=11,
        )

    # Axis formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.yaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax.set_xlabel(f"Departure date  ({dep_planet.capitalize()})", fontsize=13, color="white")
    ax.set_ylabel(f"Arrival date  ({arr_planet.capitalize()})", fontsize=13, color="white")
    ax.set_title(
        f"{dep_planet.capitalize()} → {arr_planet.capitalize()}  "
        f"Porkchop Plot  —  Total Δv  [km/s]",
        fontsize=15, color="white", pad=12,
    )
    ax.tick_params(colors="white", which="both")
    for spine in ax.spines.values():
        spine.set_color("white")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"\nPorkchop plot saved → {save_path}")
    return save_path


def plot_departure_arrival_dv(
    dep_dates: np.ndarray,
    arr_dates: np.ndarray,
    dv_dep: np.ndarray,
    dv_arr: np.ndarray,
    dv_tot: np.ndarray,
    tof_days: np.ndarray,
    dep_planet: str = "Earth",
    arr_planet: str = "Mars",
    save_path: str = "porkchop_components.png",
    dv_max: float = 20.0,
) -> str:
    """Side-by-side departure Δv and arrival Δv contour plots."""
    dep_dt = _jd_array_to_datetimes(dep_dates)
    arr_dt = _jd_array_to_datetimes(arr_dates)
    DD, AA = np.meshgrid(dep_dt, arr_dt, indexing="ij")

    levels = np.array([2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20])
    levels = levels[levels <= dv_max]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
    fig.patch.set_facecolor("#0e1117")

    for ax, data, title in [
        (ax1, dv_dep, f"Departure Δv  ({dep_planet.capitalize()})"),
        (ax2, dv_arr, f"Arrival Δv  ({arr_planet.capitalize()})"),
    ]:
        ax.set_facecolor("#0e1117")
        data_c = np.copy(data)
        data_c[data_c > dv_max] = np.nan

        cf = ax.contourf(DD, AA, data_c, levels=levels, cmap="viridis", extend="both")
        cs = ax.contour(DD, AA, data_c, levels=levels,
                        colors="white", linewidths=0.4, alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f", colors="white")

        cbar = fig.colorbar(cf, ax=ax, pad=0.02, shrink=0.85)
        cbar.set_label("Δv [km/s]", fontsize=11, color="white")
        cbar.ax.tick_params(colors="white")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.yaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        ax.set_xlabel("Departure date", fontsize=12, color="white")
        ax.set_ylabel("Arrival date", fontsize=12, color="white")
        ax.set_title(title, fontsize=14, color="white", pad=10)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

    fig.suptitle(
        f"{dep_planet.capitalize()} → {arr_planet.capitalize()}  "
        f"—  Δv Components  [km/s]",
        fontsize=16, color="white", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Component plot saved → {save_path}")
    return save_path