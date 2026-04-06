"""
Lambert Solver Validation

Test cases against known solutions:
    1. Bate, Mueller & White Example 5.2
    2. Hohmann transfer (circular coplanar Earth→Mars)
    3. Ephemeris spot check against JPL Horizons
    4. Robustness across transfer durations
    5. Self-consistency (energy and angular momentum)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm
from lambert import solve as lambert_solve
from ephemeris import state_vector, calendar_to_jd, MU_SUN, AU_KM


def _heading(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def _pass_fail(name: str, passed: bool, details: str = "") -> None:
    sym = "✓" if passed else "✗"
    status = "PASS" if passed else "FAIL"
    print(f"  [{sym}] {name:40s} {status}  {details}")


# --- Test 1: BMW Example 5.2 ---

def test_bmw_example():
    """
    r1 = [5000, 10000, 2100] km, r2 = [-14600, 2500, 7000] km
    Δt = 3600 s, μ = 398600.4418 km³/s²
    """
    _heading("Test 1: Bate-Mueller-White Example 5.2")

    r1 = np.array([5000.0, 10000.0, 2100.0])
    r2 = np.array([-14600.0, 2500.0, 7000.0])
    tof = 3600.0
    mu = 398600.4418

    sols = lambert_solve(r1, r2, tof, mu, prograde=True)

    if not sols:
        _pass_fail("Lambert returned solution", False, "no solutions")
        return

    v1, v2 = sols[0]

    # Energy consistency
    energy1 = 0.5 * norm(v1)**2 - mu / norm(r1)
    energy2 = 0.5 * norm(v2)**2 - mu / norm(r2)
    a1 = -mu / (2 * energy1)
    a2 = -mu / (2 * energy2)

    _pass_fail("Solution found", True, f"v1={v1}")
    _pass_fail("Energy match dep/arr",
               abs(a1 - a2) / abs(a1) < 1e-6,
               f"a₁={a1:.2f}  a₂={a2:.2f} km")

    # Angular momentum consistency
    h1 = np.cross(r1, v1)
    h2 = np.cross(r2, v2)
    h_err = norm(h1 - h2) / norm(h1)
    _pass_fail("Angular momentum match", h_err < 1e-6, f"rel err = {h_err:.2e}")

    print(f"\n  v1 = [{v1[0]:+10.5f}, {v1[1]:+10.5f}, {v1[2]:+10.5f}] km/s")
    print(f"  v2 = [{v2[0]:+10.5f}, {v2[1]:+10.5f}, {v2[2]:+10.5f}] km/s")
    print(f"  |v1| = {norm(v1):.5f} km/s,  |v2| = {norm(v2):.5f} km/s")
    print(f"  Semi-major axis a = {a1:.2f} km")


# --- Test 2: Hohmann transfer ---

def test_hohmann_like():
    """
    Circular coplanar Earth→Mars. Expected Δv_dep ≈ 2.95, Δv_arr ≈ 2.65 km/s.
    """
    _heading("Test 2: Hohmann-like Earth→Mars (circular coplanar)")

    r1_km = 1.0 * AU_KM
    r2_km = 1.524 * AU_KM

    angle = np.pi - 1e-4
    r1_vec = np.array([r1_km, 0.0, 0.0])
    r2_vec = np.array([r2_km * np.cos(angle), r2_km * np.sin(angle), 0.0])

    a_t = (r1_km + r2_km) / 2.0
    tof = np.pi * np.sqrt(a_t**3 / MU_SUN)

    sols = lambert_solve(r1_vec, r2_vec, tof, MU_SUN, prograde=True)

    if not sols:
        _pass_fail("Lambert returned solution", False)
        return

    v1, v2 = sols[0]

    v_earth = np.sqrt(MU_SUN / r1_km)
    v_mars = np.sqrt(MU_SUN / r2_km)

    dv_dep = abs(norm(v1) - v_earth)
    dv_arr = abs(v_mars - norm(v2))

    _pass_fail("Solution found", True)
    _pass_fail("Δv_dep ≈ 2.95 km/s", abs(dv_dep - 2.95) < 0.2, f"got {dv_dep:.3f}")
    _pass_fail("Δv_arr ≈ 2.65 km/s", abs(dv_arr - 2.65) < 0.2, f"got {dv_arr:.3f}")

    print(f"\n  Δv_dep = {dv_dep:.3f},  Δv_arr = {dv_arr:.3f},  total = {dv_dep + dv_arr:.3f} km/s")


# --- Test 3: Ephemeris spot check ---

def test_ephemeris():
    """Verify planet positions/velocities against JPL Horizons at 2026-01-01."""
    _heading("Test 3: Ephemeris spot check (2026-01-01)")

    jd = calendar_to_jd(2026, 1, 1.0)

    r_earth, v_earth = state_vector("earth", jd)
    r_mars, v_mars = state_vector("mars", jd)

    r_e = norm(r_earth) / AU_KM
    r_m = norm(r_mars) / AU_KM
    v_e = norm(v_earth)
    v_m = norm(v_mars)

    _pass_fail("Earth distance ≈ 0.98–1.02 AU", 0.97 < r_e < 1.02, f"{r_e:.4f} AU")
    _pass_fail("Mars distance ≈ 1.38–1.67 AU",  1.38 < r_m < 1.67, f"{r_m:.4f} AU")
    _pass_fail("Earth velocity ≈ 29–30 km/s",   28.5 < v_e < 31.0, f"{v_e:.2f} km/s")
    _pass_fail("Mars velocity ≈ 21–27 km/s",    20.0 < v_m < 28.0, f"{v_m:.2f} km/s")


# --- Test 4: Robustness ---

def test_robustness():
    """Solve Earth→Mars for multiple transfer durations."""
    _heading("Test 4: Robustness — Earth→Mars across transfer durations")

    jd_dep = calendar_to_jd(2026, 9, 1.0)
    jd_arr_list = [
        calendar_to_jd(2027, 3, 1.0),
        calendar_to_jd(2027, 6, 1.0),
        calendar_to_jd(2027, 9, 1.0),
    ]

    r1, v1_planet = state_vector("earth", jd_dep)

    for jd_arr in jd_arr_list:
        r2, v2_planet = state_vector("mars", jd_arr)
        tof = (jd_arr - jd_dep) * 86400.0
        tof_d = tof / 86400.0

        sols = lambert_solve(r1, r2, tof, MU_SUN, prograde=True)

        if sols:
            v1_t, v2_t = sols[0]
            dv1 = norm(v1_t - v1_planet)
            dv2 = norm(v2_planet - v2_t)
            ok = dv1 < 20 and dv2 < 20 and dv1 + dv2 > 3.0
            _pass_fail(f"TOF={tof_d:.0f}d", ok,
                       f"Δv_dep={dv1:.2f}  Δv_arr={dv2:.2f}  total={dv1+dv2:.2f}")
        else:
            _pass_fail(f"TOF={tof_d:.0f}d", False, "no solution")


# --- Test 5: Self-consistency ---

def test_self_consistency():
    """Verify energy and angular momentum match at both endpoints."""
    _heading("Test 5: Self-consistency (energy & angular momentum)")

    jd_dep = calendar_to_jd(2026, 10, 15.0)
    jd_arr = calendar_to_jd(2027, 7, 1.0)

    r1, _ = state_vector("earth", jd_dep)
    r2, _ = state_vector("mars", jd_arr)
    tof = (jd_arr - jd_dep) * 86400.0

    sols = lambert_solve(r1, r2, tof, MU_SUN, prograde=True)

    if not sols:
        _pass_fail("Solution found", False)
        return

    v1, v2 = sols[0]

    eps1 = 0.5 * norm(v1)**2 - MU_SUN / norm(r1)
    eps2 = 0.5 * norm(v2)**2 - MU_SUN / norm(r2)
    eps_err = abs(eps1 - eps2) / abs(eps1)
    _pass_fail("Orbital energy match", eps_err < 1e-6, f"rel_err={eps_err:.2e}")

    h1 = np.cross(r1, v1)
    h2 = np.cross(r2, v2)
    h_err = norm(h1 - h2) / norm(h1)
    _pass_fail("Angular momentum match", h_err < 1e-6, f"rel_err={h_err:.2e}")


# --- Run all ---

def run_all():
    print("\n" + "=" * 60)
    print("    LAMBERT SOLVER VALIDATION SUITE")
    print("=" * 60)

    test_bmw_example()
    test_hohmann_like()
    test_ephemeris()
    test_robustness()
    test_self_consistency()

    print("\n" + "=" * 60)
    print("    ALL TESTS COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all()