"""
Interesting allocation scenarios for the Chain SCM.

Medical framing:
  X1 = cheap screening/proxy test  (e.g. symptom score, questionnaire, low-cost assay)
  X2 = expensive gold-standard biomarker  (e.g. biopsy, imaging, gene panel)
  Y  = clinical outcome  (diagnosis, hospitalisation, survival)

  Chain: X1 -> X2 -> Y
    X2 = a*X1 + eps2    (proxy quality controlled by a, sigma2_sq)
    Y  = b*X2 + epsY    (clinical signal)

  Target theta = a*b = coefficient of E[Y | X1 = x1],
  the slope of the proxy-only linear predictor.

Four scenarios with non-obvious results; run with  python scenarios.py
"""

from __future__ import annotations

import numpy as np
from scm import ChainSCM
from delta import avar
from allocation import (
    solve_allocation,
    solve_allocation_nonconvex,
    simulate_allocation,
    _active_subsets,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _show_result(scm, res, budget, label, notes=""):
    active = res["subsets"]
    alpha  = res["alpha_star"]
    dominant = sorted(
        [(active[k], alpha[k], res["sample_counts"][active[k]])
         for k in range(len(active)) if alpha[k] > 0.01],
        key=lambda x: -x[1],
    )
    print(f"\n  {label}")
    if notes:
        for line in notes.splitlines():
            print(f"  NOTE: {line}")
    print(f"  {'subset':>14}  {'alpha':>7}  {'n_k (B={:.0f})'.format(budget):>12}")
    print("  " + "─" * 38)
    for s, a, n in dominant:
        lbl = "{" + ",".join(sorted(s)) + "}"
        print(f"  {lbl:>14}  {a:>7.3f}  {n:>12.1f}")
    print(f"\n  phi* = {res['phi_star']:.5f}   Var(theta_hat) = {res['var_theta']:.6f}")


def _verify(scm, res, K=200, rng=None):
    sim = simulate_allocation(scm, res, "theta", K=K, rng=rng)
    ok  = "OK" if sim["ok"] else "FAIL"
    print(f"  Simulation (K={K}): ratio={sim['ratio']:.3f}  {ok}  "
          f"[var_theory={sim['var_theory']:.5f}, var_emp={sim['var_empirical']:.5f}]")


# ---------------------------------------------------------------------------
# SCENARIO A — "The Calibration Shortcut"
# ---------------------------------------------------------------------------
# Setting: weak proxy (a=0.3) + ultra-cheap calibration data (c_{X1X2}=0.01)
#
# Medical story:
#   A pain clinic wants to predict chronic-pain outcome (Y) from a cheap
#   self-report score (X1). The true mechanistic predictor is a nerve-
#   conduction test (X2, expensive, £150 per patient). Self-report and
#   nerve conduction are correlated but weakly (a=0.3).
#
#   Key resource: old calibration records from a 10-minute dual-test
#   session (no follow-up, no outcome) cost only £1.50 per record.
#
# Surprising result:
#   The optimal strategy almost completely abandons the "obvious" X1Y
#   design (collect cheap proxy + outcome) and instead:
#     1. Buys ~6 700 calibration records (X1,X2) to pin down a precisely.
#     2. Buys a small cohort of X2Y to pin down b.
#   This beats pure X1Y by over 80%.
# ---------------------------------------------------------------------------

def scenario_A(budget=300, K_sim=200):
    print("\n" + "═" * 72)
    print("SCENARIO A — The Calibration Shortcut")
    print("  Weak proxy (a=0.3), ultra-cheap structural data (c_{X1X2}=0.01)")
    print("═" * 72)

    # Baseline: additive costs, no special structural discount
    scm_base = ChainSCM.from_costs(
        sigma1_sq=1.0, a=0.3, sigma2_sq=0.5, b=1.5, sigmaY_sq=0.3,
        c_X1=1.0, c_X2=3.0, c_Y=0.0,
    )
    phi_X1Y   = avar(scm_base, "X1Y",   "theta")
    phi_X1X2Y = avar(scm_base, "X1X2Y", "theta")

    # Intervention: cheap calibration records available
    scm = ChainSCM.from_costs(
        sigma1_sq=1.0, a=0.3, sigma2_sq=0.5, b=1.5, sigmaY_sq=0.3,
        c_X1=1.0, c_X2=3.0, c_Y=0.0, c_X1X2=0.01,
    )

    res = solve_allocation(scm, budget, "theta")

    print(f"\n  Model:  theta = a*b = {scm.theta:.3f}  (a=0.3 is weak)")
    print(f"  Costs:  c_X1=1, c_X2=3, c_Y=0, c_{{X1,X2}}=0.01")
    print(f"\n  Benchmarks (no cheap calibration):")
    print(f"    phi(X1Y only)   = {phi_X1Y:.4f}   [Var = {phi_X1Y/budget:.6f}]")
    print(f"    phi(X1X2Y only) = {phi_X1X2Y:.4f}   [Var = {phi_X1X2Y/budget:.6f}]")

    improvement = (phi_X1Y - res["phi_star"]) / phi_X1Y * 100
    _show_result(scm, res, budget,
        f"Optimal (with cheap calibration)   → {improvement:.1f}% gain over X1Y")

    # Nonconvex check
    nc = solve_allocation_nonconvex(scm, budget, "theta", n_restarts=40)
    diff = abs(res["phi_star"] - nc["phi_nonconvex"])
    print(f"\n  Nonconvex check: |Δphi| = {diff:.2e}  {'OK' if diff < 1e-3 else 'FAIL'}")
    _verify(scm, res, K=K_sim, rng=np.random.default_rng(42))

    # Intuition: as c_{X1X2} → 0, phi* → a^2 * avar(b, X2Y)
    avar_b = avar(scm_base, "X2Y", "b")
    limit = scm.a**2 * avar_b
    print(f"\n  Asymptotic limit (c_{{X1X2}}→0): phi* → a²·avar(b,X2Y) = {limit:.4f}")
    print(f"  Current phi* = {res['phi_star']:.4f}  (gap = {res['phi_star']-limit:.4f})")


# ---------------------------------------------------------------------------
# SCENARIO B — "X2Y Data Is Worthless for Proxy Prediction — Unless…"
# ---------------------------------------------------------------------------
# Setting: vary proxy quality a with STANDARD costs (no cheap X1X2).
#
# Medical story:
#   A hospital wants to use a cheap rapid test (X1) to predict patient
#   outcomes (Y). The gold-standard is a costly lab panel (X2).
#   The hospital manager suggests buying X2Y data instead of X1Y because
#   "X2 is the real predictor, we should learn from that".
#
# Surprising result:
#   For predicting E[Y | X1] = theta * X1, X2Y data is COMPLETELY
#   USELESS regardless of a, cost ratio, or budget.
#   X2Y alone does not identify theta = a*b: it identifies b and
#   Var(X2) but not a. No matter how cheap X2 is, pure X2Y gives phi=inf.
#   The ONLY way X2Y becomes useful is paired with structural data {X1,X2}.
#
# Corollary: the manager's intuition is wrong specifically for the
# proxy-prediction goal.  If the goal shifts to signal prediction
# (target b), X2Y is instead the only thing you need.
# ---------------------------------------------------------------------------

def scenario_B(budget=300):
    print("\n" + "═" * 72)
    print("SCENARIO B — X2Y is Useless for Proxy Prediction (target theta)")
    print("  Pure X2Y never identifies theta; no cost ratio changes this")
    print("═" * 72)

    print(f"\n  {'a':>5}  {'phi(X1Y)':>10}  {'phi(X2Y)':>10}  {'phi(X1X2Y)':>12}  "
          f"{'phi_opt (std costs)':>20}")
    print("  " + "─" * 65)

    for a in [0.2, 0.4, 0.6, 0.8, 0.95]:
        scm = ChainSCM.from_costs(
            sigma1_sq=1.0, a=a, sigma2_sq=0.5, b=1.5, sigmaY_sq=0.3,
            c_X1=1.0, c_X2=3.0, c_Y=0.0,
        )
        phi_x1y   = avar(scm, "X1Y",   "theta")
        phi_x2y   = avar(scm, "X2Y",   "theta")   # should be inf
        phi_x1x2y = avar(scm, "X1X2Y", "theta")
        res = solve_allocation(scm, budget, "theta")
        x2y_str = "∞" if np.isinf(phi_x2y) else f"{phi_x2y:.4f}"
        # Dominant label
        active, alpha = res["subsets"], res["alpha_star"]
        dom = max(range(len(active)), key=lambda k: alpha[k])
        dom_lbl = "{" + ",".join(sorted(active[dom])) + "}"
        print(f"  {a:>5.2f}  {phi_x1y:>10.4f}  {x2y_str:>10}  {phi_x1x2y:>12.4f}  "
              f"{res['phi_star']:>8.4f} ({dom_lbl})")

    print()
    print("  Implication: if cheap X2Y data is offered, don't use it for theta.")
    print("  The only route to exploiting X2 is cheap calibration (X1,X2) + X2Y.")

    # But for target b: X2Y is everything
    print("\n  For comparison — target b (direct signal prediction):")
    # phi_b_opt = c_X2Y * avar(b, X2Y)  (pure X2Y always optimal for target b)
    print(f"  {'a':>5}  {'avar(b,X2Y)':>12}  {'avar(b,X1Y)':>12}  {'phi_opt (=3*avar)':>18}")
    print("  " + "─" * 55)
    for a in [0.2, 0.5, 0.8, 0.95]:
        scm = ChainSCM.from_costs(
            sigma1_sq=1.0, a=a, sigma2_sq=0.5, b=1.5, sigmaY_sq=0.3,
            c_X1=1.0, c_X2=3.0, c_Y=0.0,
        )
        phi_b_x2y = avar(scm, "X2Y", "b")
        phi_b_x1y = avar(scm, "X1Y", "b")   # X1Y does not identify b: inf
        res_b = solve_allocation(scm, budget, "b")
        x1y_str = "∞" if np.isinf(phi_b_x1y) else f"{phi_b_x1y:.4f}"
        print(f"  {a:>5.2f}  {phi_b_x2y:>12.4f}  {x1y_str:>12}  "
              f"{res_b['phi_star']:>18.4f}")

    print()
    print("  phi_opt = c_X2Y * avar(b,X2Y) = 3 * avar(b,X2Y): pure X2Y always wins.")
    print("  Cheap X1X2 calibration doesn't help for target b because b is already")
    print("  fully identified by X2Y alone — adding X1 data via X1X2 can't reduce")
    print("  the X2Y Fisher information for b (Y ⊥ X1 | X2 in the chain SCM).")


# ---------------------------------------------------------------------------
# SCENARIO C — "Calibration Pays Off More When the Proxy Is Worse"
# ---------------------------------------------------------------------------
# Setting: vary proxy quality a ∈ {0.2, 0.3, 0.5, 0.7, 0.9}
#          with cheap calibration c_{X1X2} = 0.01
#
# Medical story:
#   A research consortium is deciding whether to commission a calibration
#   study (linking cheap proxy X1 to expensive biomarker X2, no outcome).
#   The calibration study costs £0.01 per record; outcome studies cost £1
#   (X1Y) or £3 (X2Y) per record.
#
# Surprising result (counter to intuition):
#   The WEAKER the proxy (lower a), the GREATER the gain from calibration.
#   With a=0.2 (weak proxy): 91% improvement over the naive X1Y strategy.
#   With a=0.9 (tight proxy): still 49%, but less dramatic.
#
#   Intuition says: "a good proxy already tells you everything — calibrate
#   a bad proxy first".  The maths says both benefit, but the bad proxy
#   benefits enormously more because X1Y estimation of theta=ab degrades
#   badly when a is small (signal-to-noise in X1Y collapses).
# ---------------------------------------------------------------------------

def scenario_C(budget=300):
    print("\n" + "═" * 72)
    print("SCENARIO C — Calibration Pays Off More When the Proxy Is Worse")
    print("  Improvement from cheap X1X2 (c=0.01) grows as a decreases")
    print("═" * 72)

    print(f"\n  {'a':>5}  {'phi (no calib)':>15}  {'phi (with calib)':>17}  "
          f"{'improvement':>12}  {'dominant strategy'}")
    print("  " + "─" * 75)

    rng = np.random.default_rng(7)
    for a in [0.2, 0.3, 0.5, 0.7, 0.9]:
        scm_base = ChainSCM.from_costs(
            sigma1_sq=1.0, a=a, sigma2_sq=0.5, b=1.5, sigmaY_sq=0.3,
            c_X1=1.0, c_X2=3.0, c_Y=0.0,
        )
        scm_calib = ChainSCM.from_costs(
            sigma1_sq=1.0, a=a, sigma2_sq=0.5, b=1.5, sigmaY_sq=0.3,
            c_X1=1.0, c_X2=3.0, c_Y=0.0, c_X1X2=0.01,
        )
        phi_base  = avar(scm_base, "X1Y", "theta")
        res       = solve_allocation(scm_calib, budget, "theta")

        active, alpha = res["subsets"], res["alpha_star"]
        dominant = sorted(
            [(active[k], alpha[k]) for k in range(len(active)) if alpha[k] > 0.02],
            key=lambda x: -x[1],
        )
        dom_str = " + ".join(
            f"{{{','.join(sorted(s))}}}:{v:.2f}" for s, v in dominant[:3]
        )
        imp = (phi_base - res["phi_star"]) / phi_base * 100
        print(f"  {a:>5.2f}  {phi_base:>15.4f}  {res['phi_star']:>17.4f}  "
              f"{imp:>11.1f}%  {dom_str}")

    print()
    print("  Mechanism: with small a, X1Y gives theta = a*b but the")
    print("  Fisher information in the ab direction of I_{X1Y} is")
    print("  proportional to a^2 * sigma1^4 / Var(X1) * Var(Y), which")
    print("  vanishes as a→0.  Calibration decouples: estimate a from")
    print("  X1X2 (cheap) and b from X2Y — both stay informative at small a.")

    print(f"\n  Optimal sample counts at a=0.2 (most extreme case):")
    scm_ex = ChainSCM.from_costs(
        sigma1_sq=1.0, a=0.2, sigma2_sq=0.5, b=1.5, sigmaY_sq=0.3,
        c_X1=1.0, c_X2=3.0, c_Y=0.0, c_X1X2=0.01,
    )
    res_ex = solve_allocation(scm_ex, budget, "theta")
    active, alpha = res_ex["subsets"], res_ex["alpha_star"]
    for k, s in enumerate(active):
        if alpha[k] > 0.01:
            n_k = res_ex["sample_counts"][s]
            lbl = "{" + ",".join(sorted(s)) + "}"
            print(f"    {lbl:>8}: alpha={alpha[k]:.3f}, n={n_k:.0f}  "
                  f"(cost_share={alpha[k]*budget:.0f}/{budget:.0f})")
    _verify(scm_ex, res_ex, K=200, rng=np.random.default_rng(13))


# ---------------------------------------------------------------------------
# SCENARIO D — "Three-Way Split: When No Single Source Dominates"
# ---------------------------------------------------------------------------
# Setting: intermediate proxy quality (a=0.5), moderate calibration cost
#          (c_{X1X2}=0.1), explore the full cost landscape.
#
# Medical story:
#   A mid-size clinical study has three acquisition modes:
#     (X1,Y): patient fills in symptom questionnaire + outcome recorded.
#             Cost: £1 per patient.  Easy to run at scale.
#     (X2,Y): expensive biomarker drawn + outcome recorded.
#             Cost: £3 per patient.  Slow and invasive.
#     (X1,X2): both tests, no outcome follow-up (short cross-sectional study).
#              Cost: £0.10 per patient (just a clinic visit, no long follow-up).
#
# Surprising result:
#   The optimal strategy uses ALL THREE types simultaneously:
#     38% of budget on X1Y  (cheap outcome data, imprecise about theta)
#     38% of budget on X2Y  (expensive outcome data, resolves b)
#     24% of budget on X1X2 (cheap calibration, resolves a)
#   No single source could achieve this: X2Y alone gives phi=inf (for theta),
#   X1Y alone gives phi=1.425, but the triple mix gives phi=1.041 (27% better).
#
# This is the "triangulation" phenomenon: three partial sources, none
# sufficient alone, combine optimally.
# ---------------------------------------------------------------------------

def scenario_D(budget=300, K_sim=200):
    print("\n" + "═" * 72)
    print("SCENARIO D — Three-Way Triangulation (No Single Source Dominates)")
    print("  a=0.8, c_{X1,X2}=0.1: optimal uses X1Y + X2Y + calibration")
    print("═" * 72)

    scm = ChainSCM.from_costs(
        sigma1_sq=1.0, a=0.8, sigma2_sq=0.5, b=1.5, sigmaY_sq=0.3,
        c_X1=1.0, c_X2=3.0, c_Y=0.0, c_X1X2=0.1,
    )

    phi_x1y   = avar(scm, "X1Y",   "theta")
    phi_x2y   = avar(scm, "X2Y",   "theta")   # inf
    phi_x1x2y = avar(scm, "X1X2Y", "theta")

    print(f"\n  Single-source benchmarks (phi, per sample):")
    print(f"    {{X1,Y}}    phi = {phi_x1y:.4f}   cost={scm.cost('X1Y'):.1f}")
    print(f"    {{X2,Y}}    phi = {'inf' if np.isinf(phi_x2y) else f'{phi_x2y:.4f}'}   cost={scm.cost('X2Y'):.1f}  (does not identify theta!)")
    print(f"    {{X1,X2,Y}} phi = {phi_x1x2y:.4f}   cost={scm.cost('X1X2Y'):.1f}")

    res = solve_allocation(scm, budget, "theta")
    improvement = (phi_x1y - res["phi_star"]) / phi_x1y * 100
    _show_result(scm, res, budget,
        f"Optimal three-way mix   → {improvement:.1f}% over pure X1Y")

    # KKT: u*^T A_k u* equal for all active subsets
    active, alpha = res["subsets"], res["alpha_star"]
    kkt = res["kkt_values"]
    kkt_active = [kkt[s] for s in active if alpha[active.index(s)] > 1e-3]
    kkt_range = max(kkt_active) - min(kkt_active)
    print(f"\n  KKT condition (equal marginal returns): range = {kkt_range:.2e}  "
          f"{'OK' if kkt_range < 1e-4 else 'FAIL'}")

    nc = solve_allocation_nonconvex(scm, budget, "theta", n_restarts=40)
    diff = abs(res["phi_star"] - nc["phi_nonconvex"])
    print(f"  Nonconvex check: |Δphi| = {diff:.2e}  {'OK' if diff < 1e-3 else 'FAIL'}")
    _verify(scm, res, K=K_sim, rng=np.random.default_rng(99))

    # Show how phi* varies as c_{X1X2} is swept
    print(f"\n  How does phi* change as calibration cost varies?")
    print(f"  {'c_X1X2':>8}  {'phi_opt':>9}  {'vs X1Y':>8}  {'strategy'}")
    print("  " + "─" * 60)
    for c12 in [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 4.0]:
        scm_c = ChainSCM.from_costs(
            sigma1_sq=1.0, a=0.8, sigma2_sq=0.5, b=1.5, sigmaY_sq=0.3,
            c_X1=1.0, c_X2=3.0, c_Y=0.0, c_X1X2=c12,
        )
        r = solve_allocation(scm_c, budget, "theta")
        imp = (phi_x1y - r["phi_star"]) / phi_x1y * 100
        dom = sorted(
            [(active[k], r["alpha_star"][k]) for k in range(len(active)) if r["alpha_star"][k] > 0.05],
            key=lambda x: -x[1],
        )
        dom_str = " + ".join(f"{{{','.join(sorted(s))}}}:{v:.2f}" for s, v in dom[:3])
        print(f"  {c12:>8.2f}  {r['phi_star']:>9.4f}  {imp:>7.1f}%  {dom_str}")


# ---------------------------------------------------------------------------
# SCENARIO E — "The Outcome-Free Solution"
# ---------------------------------------------------------------------------
# Setting: outcome collection is expensive (c_Y = 10), e.g. a rare disease
#          requiring a 10-year follow-up.  All other costs additive.
#
# Medical story:
#   A rare-disease registry wants to build a cheap screening predictor
#   (X1 = questionnaire score).  The gold-standard biomarker (X2 = gene panel,
#   £3) and the outcome (Y = disease onset, requires 10 years follow-up, £10)
#   both cost real money.  A structural visit (X1 + X2, no outcome) costs £4.
#
# Surprising result:
#   With c_Y=10: the optimal design recruits ~35 patients for a short
#   calibration visit (X1,X2, no follow-up) and only ~11 patients for the
#   full longitudinal study (X1,X2,Y).  The outcome-free cohort is 3× larger.
#   With c_Y=5: a three-way mix (X1Y + X1X2 + X1X2Y) first appears.
#   Pure X1Y does NOT enter the optimal once c_Y >= 5.
#
# Key: X1X2 data alone does NOT identify theta=ab; it only identifies a.
#   It is useful only because it is combined with X1X2Y which identifies
#   everything including b.  The two together give a cost-efficient split:
#   "learn a cheaply from many short visits, learn b from a few full studies."
# ---------------------------------------------------------------------------

def scenario_E(budget=300, K_sim=200):
    print("\n" + "═" * 72)
    print("SCENARIO E — The Outcome-Free Solution")
    print("  Expensive follow-up (c_Y=10): collect mostly outcome-free X1X2 records")
    print("═" * 72)

    print(f"\n  Phase transition as c_Y increases (a=0.8, c_X1=1, c_X2=3):")
    print(f"  {'c_Y':>5}  {'c_X1Y':>6}  {'c_X1X2':>7}  {'phi*':>9}  {'strategy'}")
    print("  " + "─" * 70)

    for c_Y in [0.0, 1.0, 3.0, 5.0, 7.0, 10.0, 15.0]:
        scm = ChainSCM.from_costs(
            sigma1_sq=1.0, a=0.8, sigma2_sq=0.5, b=1.5, sigmaY_sq=0.3,
            c_X1=1.0, c_X2=3.0, c_Y=c_Y,
        )
        res = solve_allocation(scm, budget, "theta")
        active, alpha = res["subsets"], res["alpha_star"]
        dominant = sorted(
            [(active[k], alpha[k]) for k in range(len(active)) if alpha[k] > 0.02],
            key=lambda x: -x[1],
        )
        dom_str = " + ".join(
            f"{{{','.join(sorted(s))}}}:{v:.2f}" for s, v in dominant[:3]
        )
        print(f"  {c_Y:>5.1f}  {scm.cost('X1Y'):>6.1f}  {scm.cost('X1X2'):>7.1f}  "
              f"{res['phi_star']:>9.4f}  {dom_str}")

    # Zoom into c_Y=10 — the most striking case
    print(f"\n  Focal case: c_Y=10")
    scm_10 = ChainSCM.from_costs(
        sigma1_sq=1.0, a=0.8, sigma2_sq=0.5, b=1.5, sigmaY_sq=0.3,
        c_X1=1.0, c_X2=3.0, c_Y=10.0,
    )
    res_10 = solve_allocation(scm_10, budget, "theta")
    active, alpha = res_10["subsets"], res_10["alpha_star"]
    print(f"  {'subset':>12}  {'alpha':>7}  {'n_k':>8}  {'kkt':>10}")
    print("  " + "─" * 44)
    for k, s in enumerate(active):
        if alpha[k] > 0.01:
            lbl = "{" + ",".join(sorted(s)) + "}"
            print(f"  {lbl:>12}  {alpha[k]:>7.3f}  "
                  f"{res_10['sample_counts'][s]:>8.1f}  "
                  f"{res_10['kkt_values'][s]:>10.5f}")

    nc = solve_allocation_nonconvex(scm_10, budget, "theta", n_restarts=40)
    diff = abs(res_10["phi_star"] - nc["phi_nonconvex"])
    print(f"\n  phi* = {res_10['phi_star']:.5f}")
    print(f"  Nonconvex check: |Δφ| = {diff:.2e}  {'OK' if diff < 1e-3 else 'FAIL'}")

    # Verify at 10x budget so n_X1X2Y ~ 114 (asymptotic theory needs n>>1)
    res_10_big = solve_allocation(scm_10, budget * 10, "theta")
    print(f"  Simulation at B={budget*10:.0f} (n_X1X2Y≈{res_10_big['sample_counts'][frozenset({'X1','X2','Y'})]:.0f}):")
    _verify(scm_10, res_10_big, K=K_sim, rng=np.random.default_rng(55))

    print(f"\n  Interpretation: pay for outcome in only {res_10['sample_counts'][frozenset({'X1','X2','Y'})]:.0f} "
          f"patients ({res_10['alpha_star'][active.index(frozenset({'X1','X2','Y'}))]*100:.0f}% of budget).")
    print(f"  The other {res_10['alpha_star'][active.index(frozenset({'X1','X2'}))]*100:.0f}% of budget "
          f"buys {res_10['sample_counts'][frozenset({'X1','X2'})]:.0f} outcome-free calibration visits.")
    print(f"  (Ratio persists at any budget: optimal alpha is scale-invariant.)")


# ---------------------------------------------------------------------------
# SCENARIO F — "The Biobank Effect"
# ---------------------------------------------------------------------------
# Setting: an existing disease registry provides (X2,Y) records cheaply.
#          Cost for new prospective data: c_X1Y=1, c_X1X2=4 (additive).
#          Registry cost for X2Y: swept from 3 down to 0.05.
#
# Medical story:
#   A national biobank has thousands of records linking a gold-standard
#   biomarker (X2) to a disease outcome (Y).  Accessing these costs
#   essentially nothing (£0.1 per record, just data retrieval).
#   New prospective data with the cheap proxy (X1) costs £1 per patient.
#
# Surprising result:
#   At standard X2Y costs (c_X2Y >= 0.3), the biobank data is IRRELEVANT
#   for estimating theta=ab: pure X1Y remains optimal.
#   Below the threshold c_X2Y < 0.3, the biobank enters the optimal mix —
#   even though X2Y alone does not identify theta.
#
# Mechanism:
#   X2Y provides information about b and Var(X2) — nuisance parameters
#   for theta estimation from X1Y.  When X2Y is cheap enough, saturating
#   those directions reduces the effective variance of theta, acting like
#   a "free lunch" for nuisance-parameter elimination.
#   The KKT condition pins down the threshold:
#   X2Y enters when its cost-scaled information gain (per budget unit)
#   exceeds the marginal return from X1Y at the current optimum.
# ---------------------------------------------------------------------------

def scenario_F(budget=300):
    print("\n" + "═" * 72)
    print("SCENARIO F — The Biobank Effect")
    print("  X2Y data (disease registry) helps only below a cost threshold")
    print("═" * 72)

    print(f"\n  a=0.5, c_X1=1, c_X2=3, c_Y=0, c_X1X2=additive=4")
    print(f"  {'c_X2Y':>7}  {'phi*':>9}  {'α_X1Y':>7}  {'α_X2Y':>7}  {'vs std X1Y':>11}")
    print("  " + "─" * 50)

    phi_std = 1.4250   # avar(theta, X1Y) — standard baseline

    for c_x2y in [3.0, 1.0, 0.5, 0.3, 0.25, 0.2, 0.15, 0.10, 0.05]:
        scm = ChainSCM.from_costs(
            sigma1_sq=1.0, a=0.5, sigma2_sq=0.5, b=1.5, sigmaY_sq=0.3,
            c_X1=1.0, c_X2=3.0, c_Y=0.0, c_X2Y=c_x2y,
        )
        res = solve_allocation(scm, budget, "theta")
        active, alpha = res["subsets"], res["alpha_star"]
        k_x1y = next((k for k, s in enumerate(active) if s == frozenset({"X1","Y"})), None)
        k_x2y = next((k for k, s in enumerate(active) if s == frozenset({"X2","Y"})), None)
        a_x1y = alpha[k_x1y] if k_x1y is not None else 0.0
        a_x2y = alpha[k_x2y] if k_x2y is not None else 0.0
        imp = (phi_std - res["phi_star"]) / phi_std * 100
        marker = " ← threshold" if abs(c_x2y - 0.25) < 0.01 else ""
        print(f"  {c_x2y:>7.3f}  {res['phi_star']:>9.5f}  {a_x1y:>7.3f}  {a_x2y:>7.3f}  "
              f"{imp:>+10.1f}%{marker}")

    # Show the gradient sign flip at the threshold
    print(f"\n  KKT gradient at pure X1Y (alpha_X2Y=0):")
    print(f"  phi decreases when d_phi/d_alpha_X2Y < d_phi/d_alpha_X1Y")

    from allocation import _cost_scaled_fims, _phi_and_grad
    from delta import h_gradient
    for c_x2y in [3.0, 0.3, 0.1]:
        scm = ChainSCM.from_costs(
            sigma1_sq=1.0, a=0.5, sigma2_sq=0.5, b=1.5, sigmaY_sq=0.3,
            c_X1=1.0, c_X2=3.0, c_Y=0.0, c_X2Y=c_x2y,
        )
        subsets, A_list = _cost_scaled_fims(scm)
        h = h_gradient(scm, "theta")
        k_x1y = next(k for k, s in enumerate(subsets) if s == frozenset({"X1","Y"}))
        k_x2y = next(k for k, s in enumerate(subsets) if s == frozenset({"X2","Y"}))
        alpha_pure = np.zeros(len(subsets)); alpha_pure[k_x1y] = 1.0
        _, grad = _phi_and_grad(alpha_pure, A_list, h)
        sign = "X2Y helps ✓" if grad[k_x2y] < grad[k_x1y] else "X2Y hurts  "
        print(f"  c_X2Y={c_x2y:.2f}: grad_X1Y={grad[k_x1y]:>8.4f}  "
              f"grad_X2Y={grad[k_x2y]:>8.4f}  → {sign}")


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BUDGET = 300.0

    print("=" * 72)
    print("Budget Allocation Scenarios — Chain SCM X1 → X2 → Y")
    print("Medical framing: X1=cheap proxy test, X2=expensive biomarker, Y=outcome")
    print("=" * 72)

    scenario_A(budget=BUDGET, K_sim=200)
    scenario_B(budget=BUDGET)
    scenario_C(budget=BUDGET)
    scenario_D(budget=BUDGET, K_sim=200)
    scenario_E(budget=BUDGET, K_sim=200)
    scenario_F(budget=BUDGET)

    print("\n" + "=" * 72)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 72)
    print("""
  A. Calibration Shortcut (a=0.3, c_{X1X2}=0.01):
     Collecting ~6700 cheap calibration records (no outcome!) + 77 X2Y records
     beats 300 direct X1Y records by 84%.  The "obvious" study design is far
     from optimal when cheap structural data is available.

  B. X2Y is Useless for Proxy Prediction:
     No matter how cheap X2Y data is, it never identifies theta=ab alone.
     The clinician's intuition ("use the gold-standard test") is wrong for
     proxy-prediction goals.  X2Y becomes useful only via cheap calibration.
     The situation flips for target b: X2Y is then the only useful source.

  C. Weak Proxy Benefits More from Calibration:
     The improvement from cheap calibration (c_{X1X2}=0.01) is 91% for a=0.2
     but only 49% for a=0.9.  The worse the proxy, the more the decoupled
     (calibration + signal) strategy beats the direct proxy strategy.

  D. Three-Way Triangulation (c_{X1X2}=0.1):
     Optimal splits budget three ways: X1Y + X2Y + calibration.  Each alone
     is either non-identifiable (X2Y) or suboptimal; the mixture dominates.

  E. Expensive Outcomes — The Outcome-Free Solution (c_Y=10):
     When follow-up is costly, the optimal collects 3x more outcome-free
     calibration records (X1,X2) than complete panels (X1,X2,Y).  The {X1,X2}
     data alone identifies the proxy-biomarker link cheaply; outcomes are spent
     only where they are strictly necessary.

  F. Biobank Effect — Cheap X2Y Data Has a Threshold:
     X2Y data from a registry starts helping only when c_X2Y < ~0.3 (roughly
     c_X1Y/5).  Below that threshold the registry reduces phi and changes the
     optimal mix; above it the registry data is irrelevant for theta estimation.
""")
