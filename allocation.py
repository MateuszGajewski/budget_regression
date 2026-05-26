"""
Fisher-optimal budget allocation for the Chain SCM.

Given a ChainSCM (with costs for all 7 non-empty subsets) and a budget B,
finds the allocation α* over active subsets (those with c_k > 0) that
minimises the asymptotic variance of the MLE estimator of the target:

    φ(α) = h^T M(α)^{-1} h,     M(α) = Σ_k (α_k / c_k) · I_k(ψ)
    α* = argmin_{α ∈ Δ_K} φ(α)
    Var(θ̂) ≈ φ(α*) / B,    Excess MSE ≈ σ₁² · φ(α*) / B

Three sanity checks in __main__:
  1. Convex (SLSQP) vs Nonconvex (Nelder-Mead) solver agree on φ*.
  2. K=200 MLE simulations at α* match the theoretical Var(θ̂).
  3. Sparse grid search finds no empirically better allocation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize

from scm import ChainSCM, ALL_SUBSETS
from fisher import fisher_information, N_PARAMS
from delta import h_gradient, target_value
from mle import mle_fit


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _active_subsets(scm: ChainSCM) -> List:
    """ALL_SUBSETS filtered to those with scm.cost(s) > 0."""
    return [s for s in ALL_SUBSETS if scm.cost(s) > 0]


def _cost_scaled_fims(scm: ChainSCM) -> Tuple[List, List[np.ndarray]]:
    """Return (subsets, A_list) where A_k = I_k(ψ) / c_k."""
    subsets = _active_subsets(scm)
    A_list = [fisher_information(scm, s) / scm.cost(s) for s in subsets]
    return subsets, A_list


def _M_matrix(alpha: np.ndarray, A_list: List[np.ndarray]) -> np.ndarray:
    """M(α) = Σ_k α_k · A_k."""
    M = np.zeros((N_PARAMS, N_PARAMS))
    for k, A in enumerate(A_list):
        M += alpha[k] * A
    return M


def _phi_and_grad(
    alpha: np.ndarray,
    A_list: List[np.ndarray],
    h: np.ndarray,
    rcond: float = 1e-8,
) -> Tuple[float, np.ndarray]:
    """φ(α) = h^T M(α)^{-1} h and gradient ∂φ/∂α_k = −u^T A_k u.

    Falls back to pseudoinverse for rank-deficient M (boundary allocations).
    Returns (1e15, zeros) when h is outside col(M), i.e. target non-identifiable.
    """
    K = len(A_list)
    M = _M_matrix(alpha, A_list)

    # Try direct solve; fall back to pseudoinverse for rank-deficient M.
    # We always check the residual because np.linalg.solve can silently
    # return inaccurate results for near-singular M without raising LinAlgError.
    solved_ok = False
    try:
        u = np.linalg.solve(M, h)
        residual = np.linalg.norm(M @ u - h)
        if residual <= 1e-6 * (np.linalg.norm(h) + 1.0):
            solved_ok = True
    except np.linalg.LinAlgError:
        pass

    if not solved_ok:
        # Singular or ill-conditioned M: use pseudoinverse and check identifiability
        u = np.linalg.pinv(M, rcond=rcond) @ h
        residual = np.linalg.norm(M @ u - h)
        if residual > 1e-6 * (np.linalg.norm(h) + 1.0):
            return 1e15, np.zeros(K)  # h not in col(M) → φ = ∞

    phi_val = float(h @ u)
    if not np.isfinite(phi_val) or phi_val < 0:
        return 1e15, np.zeros(K)

    grad = np.array([-float(u @ A @ u) for A in A_list])
    return phi_val, grad


# ---------------------------------------------------------------------------
# Convex solver
# ---------------------------------------------------------------------------

def solve_allocation(
    scm: ChainSCM,
    budget: float,
    target: str = "theta",
    n_restarts: int = 10,       # kept for API compatibility; unused by CVXPY
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """Fisher-optimal allocation via CVXPY (disciplined convex programming).

    Minimises  φ(α) = h^T M(α)^{-1} h  over the simplex Δ_K using the
    CVXPY `matrix_frac` atom, which is recognised as convex and dispatched
    to an interior-point solver (CLARABEL).  A single solve is guaranteed
    to reach the global optimum; no restarts are needed.

    Parameters
    ----------
    scm    : ChainSCM — provides ψ and per-subset costs
    budget : total measurement budget B
    target : "theta" or "b"

    Returns
    -------
    dict with keys:
        subsets       : list[frozenset]  active acquisition types (K entries)
        alpha_star    : ndarray (K,)
        sample_counts : dict frozenset → float  n_k = α_k · B / c_k
        phi_star      : minimised h^T M(α*)^{-1} h
        var_theta     : phi_star / budget
        excess_mse    : σ₁² · var_theta
        kkt_values    : dict frozenset → u*^T A_k u*  (equal for active k at KKT)
    """
    subsets, A_list = _cost_scaled_fims(scm)
    h = h_gradient(scm, target)
    K = len(subsets)

    alpha = cp.Variable(K, nonneg=True)

    # M(α) = Σ_k α_k · A_k  — a positive-semidefinite matrix expression
    M_expr = sum(alpha[k] * A_list[k] for k in range(K))

    # matrix_frac(h, M) = h^T M^{-1} h — convex in M (and hence in α)
    objective = cp.Minimize(cp.matrix_frac(h, M_expr))
    constraints = [cp.sum(alpha) == 1]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ("optimal", "optimal_inaccurate") or alpha.value is None:
        raise RuntimeError(f"CVXPY solve failed: status={prob.status!r}")

    best_alpha = np.clip(alpha.value, 0.0, None)
    best_alpha /= best_alpha.sum()
    best_val = float(prob.value)

    costs = [scm.cost(s) for s in subsets]
    sample_counts = {s: best_alpha[k] * budget / costs[k] for k, s in enumerate(subsets)}

    # KKT marginal values  u*^T A_k u*,  u* = M(α*)^{-1} h
    M_star = _M_matrix(best_alpha, A_list)
    try:
        u_star = np.linalg.solve(M_star, h)
    except np.linalg.LinAlgError:
        u_star = np.linalg.pinv(M_star) @ h
    kkt_values = {s: float(u_star @ A_list[k] @ u_star) for k, s in enumerate(subsets)}

    return {
        "subsets"      : subsets,
        "alpha_star"   : best_alpha,
        "sample_counts": sample_counts,
        "phi_star"     : best_val,
        "var_theta"    : best_val / budget,
        "excess_mse"   : scm.sigma1_sq * best_val / budget,
        "kkt_values"   : kkt_values,
    }


# ---------------------------------------------------------------------------
# Nonconvex sanity check
# ---------------------------------------------------------------------------

def solve_allocation_nonconvex(
    scm: ChainSCM,
    budget: float,
    target: str = "theta",
    n_restarts: int = 40,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """Nelder-Mead on softmax-parameterised simplex (gradient-free check).

    α = softmax(z),  z ∈ R^K  (automatically satisfies α ≥ 0, Σα = 1).
    Multiple random starts from N(0,I_K).
    """
    if rng is None:
        rng = np.random.default_rng(1)

    subsets, A_list = _cost_scaled_fims(scm)
    h = h_gradient(scm, target)
    K = len(subsets)

    def _softmax(z: np.ndarray) -> np.ndarray:
        e = np.exp(z - z.max())
        return e / e.sum()

    def objective(z: np.ndarray) -> float:
        phi_val, _ = _phi_and_grad(_softmax(z), A_list, h)
        return phi_val

    best_val = np.inf
    best_alpha = np.ones(K) / K

    for _ in range(n_restarts):
        z0 = rng.normal(0.0, 1.0, K)
        res = minimize(objective, z0, method="Nelder-Mead",
                       options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10})
        if np.isfinite(res.fun) and res.fun < best_val:
            best_val = res.fun
            best_alpha = _softmax(res.x)

    return {
        "subsets"        : subsets,
        "alpha_nonconvex": best_alpha,
        "phi_nonconvex"  : best_val,
    }


# ---------------------------------------------------------------------------
# Simulation verification
# ---------------------------------------------------------------------------

def simulate_allocation(
    scm: ChainSCM,
    allocation_result: Dict,
    target: str = "theta",
    K: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """Run K MLE fits from the optimal allocation; compare Var(θ̂) to theory.

    Returns
    -------
    dict: var_theory, var_empirical, ratio, ok (ratio ∈ [0.8, 1.2])
    """
    if rng is None:
        rng = np.random.default_rng(2)

    sample_counts = allocation_result["sample_counts"]
    var_theory = allocation_result["var_theta"]
    true_val = target_value(scm, target)

    # Round; keep subsets with at least 2 samples
    rounded = {s: max(2, round(n)) for s, n in sample_counts.items() if round(n) >= 1}

    estimates = []
    for _ in range(K):
        data_dict = {s: scm.sample(s, n, rng=rng) for s, n in rounded.items()}
        scm_fit = mle_fit(scm, data_dict, n_restarts=1, rng=rng)
        est = scm_fit.a * scm_fit.b if target == "theta" else scm_fit.b
        estimates.append(est)

    estimates = np.array(estimates)
    var_emp = float(np.var(estimates))
    ratio = var_emp / var_theory if var_theory > 0 else np.inf

    return {
        "var_theory"   : var_theory,
        "var_empirical": var_emp,
        "ratio"        : ratio,
        "ok"           : 0.8 <= ratio <= 1.2,
        "theta_mean"   : float(np.mean(estimates)),
        "theta_true"   : true_val,
    }


# ---------------------------------------------------------------------------
# Grid-search verification
# ---------------------------------------------------------------------------

def grid_search_allocation(
    scm: ChainSCM,
    budget: float,
    target: str = "theta",
    grid_size: int = 5,
    K_sim: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Evaluate a sparse grid of allocations via sample → MLE → θ̂.

    Grid construction:
      K=1 : single point [1]
      K=2 : uniform 1-D grid of grid_size+1 points
      K=3 : regular simplex grid (~(grid_size+1)(grid_size+2)/2 points)
      K>3 : grid_size² random Dirichlet(1,...,1) draws

    Each entry:
        alpha         : allocation vector (K,)
        phi_theory    : h^T M(α)^{-1} h  (np.inf if non-identifiable)
        var_empirical : empirical Var(θ̂) from K_sim MLE runs (None if skipped)
        n_sim         : simulations actually run
    """
    if rng is None:
        rng = np.random.default_rng(3)

    subsets, A_list = _cost_scaled_fims(scm)
    h = h_gradient(scm, target)
    K_act = len(subsets)
    costs = [scm.cost(s) for s in subsets]

    # Build grid
    grid_alphas: List[np.ndarray] = []
    if K_act == 1:
        grid_alphas = [np.array([1.0])]
    elif K_act == 2:
        for i in range(grid_size + 1):
            a = i / grid_size
            grid_alphas.append(np.array([a, 1.0 - a]))
    elif K_act == 3:
        for i in range(grid_size + 1):
            for j in range(grid_size + 1 - i):
                a1, a2 = i / grid_size, j / grid_size
                grid_alphas.append(np.array([a1, a2, max(0.0, 1.0 - a1 - a2)]))
    else:
        for _ in range(grid_size * grid_size):
            d = rng.exponential(1.0, K_act)
            grid_alphas.append(d / d.sum())

    results = []
    for alpha in grid_alphas:
        phi_val, _ = _phi_and_grad(alpha, A_list, h)

        if phi_val >= 1e14:
            results.append({
                "alpha": alpha, "phi_theory": np.inf,
                "var_empirical": None, "n_sim": 0,
            })
            continue

        rounded = {
            s: max(2, round(alpha[k] * budget / costs[k]))
            for k, s in enumerate(subsets)
            if round(alpha[k] * budget / costs[k]) >= 2
        }
        if not rounded:
            results.append({
                "alpha": alpha, "phi_theory": phi_val,
                "var_empirical": None, "n_sim": 0,
            })
            continue

        ests = []
        for _ in range(K_sim):
            data_dict = {s: scm.sample(s, n, rng=rng) for s, n in rounded.items()}
            scm_fit = mle_fit(scm, data_dict, n_restarts=1, rng=rng)
            ests.append(scm_fit.a * scm_fit.b if target == "theta" else scm_fit.b)

        results.append({
            "alpha"        : alpha,
            "phi_theory"   : phi_val,
            "var_empirical": float(np.var(ests)),
            "n_sim"        : K_sim,
        })

    return results


# ---------------------------------------------------------------------------
# __main__: three verification checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TRUE_SCM = ChainSCM.from_costs(
        sigma1_sq=1.0, a=0.8, sigma2_sq=0.5, b=1.5, sigmaY_sq=0.3,
        c_X1=1.0, c_X2=3.0, c_Y=0.0,
    )
    BUDGET = 300.0
    TARGET = "theta"

    active = _active_subsets(TRUE_SCM)
    labels = ["{" + ",".join(sorted(s)) + "}" for s in active]
    K_act = len(active)

    print("=" * 72)
    print("Fisher-Optimal Budget Allocation — Chain SCM")
    print(f"ψ  = (σ₁²={TRUE_SCM.sigma1_sq}, a={TRUE_SCM.a}, σ₂²={TRUE_SCM.sigma2_sq}, "
          f"b={TRUE_SCM.b}, σY²={TRUE_SCM.sigmaY_sq})")
    print(f"θ  = ab = {TRUE_SCM.theta:.4f}")
    print(f"Budget B = {BUDGET},  target = {TARGET}")
    print(f"Active subsets (K={K_act}, c_Y=0 → {{Y}} excluded):")
    for s, lbl in zip(active, labels):
        print(f"    {lbl:<16}  cost = {TRUE_SCM.cost(s):.1f}")
    print("=" * 72)

    # ── CHECK 1: Convex vs nonconvex ─────────────────────────────────────────
    print("\nCHECK 1 — Convex (SLSQP) vs Nonconvex (Nelder-Mead)")
    print("─" * 72)

    res_conv  = solve_allocation(TRUE_SCM, BUDGET, TARGET)
    res_nconv = solve_allocation_nonconvex(TRUE_SCM, BUDGET, TARGET)

    print(f"\n{'subset':>16}  {'α_convex':>10}  {'α_noncvx':>10}  "
          f"{'n_k':>8}  {'kkt_val':>10}")
    print("─" * 62)
    for k, s in enumerate(active):
        lbl = labels[k]
        n_k = res_conv["sample_counts"][s]
        kkt = res_conv["kkt_values"].get(s, float("nan"))
        # dim active or not
        marker = " *" if res_conv["alpha_star"][k] > 1e-3 else "  "
        print(f"{lbl:>16}{marker}  {res_conv['alpha_star'][k]:>10.5f}  "
              f"{res_nconv['alpha_nonconvex'][k]:>10.5f}  "
              f"{n_k:>8.1f}  {kkt:>10.6f}")

    diff = abs(res_conv["phi_star"] - res_nconv["phi_nonconvex"])
    print(f"\n  φ_convex    = {res_conv['phi_star']:.8f}")
    print(f"  φ_nonconvex = {res_nconv['phi_nonconvex']:.8f}")
    print(f"  |Δφ|        = {diff:.2e}  {'OK' if diff < 1e-3 else 'FAIL'}")
    print(f"\n  Var(θ̂) ≈ φ*/B = {res_conv['var_theta']:.8f}")
    print(f"  Excess MSE   = σ₁²·Var(θ̂) = {res_conv['excess_mse']:.8f}")

    print("\n  KKT check: u*^T A_k u* should be equal for active subsets.")
    kkt_active = [v for s, v in res_conv["kkt_values"].items()
                  if res_conv["alpha_star"][active.index(s)] > 1e-3]
    if kkt_active:
        kkt_range = max(kkt_active) - min(kkt_active)
        print(f"  KKT range (active subsets): {kkt_range:.2e}  "
              f"{'OK' if kkt_range < 1e-4 else 'FAIL'}")

    # ── CHECK 2: Simulation CRB ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CHECK 2 — Simulation CRB at optimal α*  (K=200 repetitions)")
    print("─" * 72)

    sim = simulate_allocation(TRUE_SCM, res_conv, TARGET, K=200)
    print(f"\n  var_theory    = {sim['var_theory']:.8f}")
    print(f"  var_empirical = {sim['var_empirical']:.8f}")
    print(f"  ratio         = {sim['ratio']:.4f}  {'OK' if sim['ok'] else 'FAIL'}")
    print(f"  θ_true        = {sim['theta_true']:.6f}")
    print(f"  θ_mean_hat    = {sim['theta_mean']:.6f}")

    # ── CHECK 3: Grid search ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CHECK 3 — Grid search (grid_size=5, K_sim=10 per point)")
    print("─" * 72)

    grid = grid_search_allocation(TRUE_SCM, BUDGET, TARGET, grid_size=5, K_sim=10)
    grid_run = [r for r in grid if r["var_empirical"] is not None]
    grid_run.sort(key=lambda r: r["phi_theory"])

    opt_var = res_conv["var_theta"]

    print(f"\n  Optimal var_theory = {opt_var:.8f}")
    print(f"\n  {'phi_theory':>12}  {'var_emp':>12}  {'ratio/opt':>10}  {'top subsets'}")
    print("  " + "─" * 65)
    for r in grid_run[:25]:
        ratio_str = f"{r['var_empirical']/opt_var:.3f}" if r["var_empirical"] else "—"
        # Find top-2 subset labels by alpha weight
        top_idx = np.argsort(r["alpha"])[::-1][:2]
        top_lbl = " + ".join(
            f"{labels[i]}={r['alpha'][i]:.2f}" for i in top_idx if r["alpha"][i] > 0.01
        )
        print(f"  {r['phi_theory']:>12.6f}  {r['var_empirical']:>12.8f}  "
              f"{ratio_str:>10}  {top_lbl}")

    # Empirical optimality check
    n_beaten = sum(
        1 for r in grid_run
        if r["var_empirical"] is not None and r["var_empirical"] < opt_var * 0.85
    )
    print(f"\n  Grid points with var_emp < 85% of optimal theory: {n_beaten}  "
          f"{'OK' if n_beaten == 0 else 'NOTE: could be noise or better allocation'}")

    # ── SCENARIO 2: cheap {X1,X2} ────────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print("SCENARIO 2 — Cheap {X1,X2} (cost=0.1)")
    print("Setting: c_X1=1, c_X2=3, c_Y=0, c_X1X2=0.1  (all others additive)")
    print("=" * 72)
    BUDGET = 100

    CHEAP_SCM = ChainSCM.from_costs(
        sigma1_sq=1.0, a=0.8, sigma2_sq=0.5, b=1.5, sigmaY_sq=0.3,
        c_X1=1.0, c_X2=1.0, c_Y=0.0,
        c_X1X2=0.1,
    )

    active2 = _active_subsets(CHEAP_SCM)
    labels2 = ["{" + ",".join(sorted(s)) + "}" for s in active2]

    print(f"\nActive subsets and costs:")
    for s, lbl in zip(active2, labels2):
        print(f"    {lbl:<16}  cost = {CHEAP_SCM.cost(s):.1f}")

    print("\nOptimal allocation:\n")
    print(f"  {'subset':>16}  {'alpha':>8}  {'n_k':>10}  {'kkt_val':>10}")
    print("  " + "─" * 50)

    res2 = solve_allocation(CHEAP_SCM, BUDGET, TARGET)
    for k, s in enumerate(active2):
        marker = " *" if res2["alpha_star"][k] > 1e-3 else "  "
        print(f"  {labels2[k]:>16}{marker}  {res2['alpha_star'][k]:>8.4f}  "
              f"{res2['sample_counts'][s]:>10.1f}  "
              f"{res2['kkt_values'][s]:>10.6f}")

    ref_phi = TRUE_SCM.cost(frozenset({"X1","Y"}))   # not quite; use avar
    from delta import avar
    phi_ref = avar(TRUE_SCM, "X1Y", TARGET)
    improvement = (phi_ref - res2["phi_star"]) / phi_ref * 100

    print(f"\n  φ*         = {res2['phi_star']:.8f}")
    print(f"  Var(θ̂)    = {res2['var_theta']:.8f}")
    print(f"  Excess MSE = {res2['excess_mse']:.8f}")
    print(f"\n  Reference: pure {{X1,Y}}  φ={phi_ref:.6f}  Var(θ̂)={phi_ref/BUDGET:.8f}")
    print(f"  Improvement over pure {{X1,Y}}: {improvement:.1f}%")

    res2_nc = solve_allocation_nonconvex(CHEAP_SCM, BUDGET, TARGET)
    diff2 = abs(res2["phi_star"] - res2_nc["phi_nonconvex"])
    print(f"\nNonconvex check (Nelder-Mead):")
    print(f"  φ_nonconvex = {res2_nc['phi_nonconvex']:.8f}  |Δφ| = {diff2:.2e}  "
          f"{'OK' if diff2 < 1e-3 else 'FAIL'}")

    sim2 = simulate_allocation(CHEAP_SCM, res2, TARGET, K=200)
    print(f"\nSimulation CRB (K=200 repetitions):")
    print(f"  var_theory    = {sim2['var_theory']:.8f}")
    print(f"  var_empirical = {sim2['var_empirical']:.8f}")
    print(f"  ratio         = {sim2['ratio']:.4f}  {'OK' if sim2['ok'] else 'FAIL'}")
    print(f"  θ_true={sim2['theta_true']:.4f}  θ_mean_hat={sim2['theta_mean']:.4f}")

    grid2 = grid_search_allocation(CHEAP_SCM, BUDGET, TARGET, grid_size=5, K_sim=50)
    grid2_run = sorted(
        [r for r in grid2 if r["var_empirical"] is not None],
        key=lambda r: r["phi_theory"],
    )
    opt_var2 = res2["var_theta"]
    print(f"\nGrid search (grid_size=5, K_sim=10):\n")
    print(f"  {'phi_theory':>12}  {'var_emp':>12}  {'ratio/opt':>10}  top subsets")
    print("  " + "─" * 70)
    for r in grid2_run[:20]:
        top_idx = np.argsort(r["alpha"])[::-1][:2]
        top_lbl = " + ".join(
            f"{labels2[i]}={r['alpha'][i]:.2f}" for i in top_idx if r["alpha"][i] > 0.01
        )
        print(f"  {r['phi_theory']:>12.4f}  {r['var_empirical']:>12.8f}  "
              f"{r['var_empirical']/opt_var2:>10.3f}  {top_lbl}")

    n_beaten2 = sum(
        1 for r in grid2_run if r["var_empirical"] < opt_var2 * 0.85
    )
    print(f"\n  Grid points beating 85% of optimal: {n_beaten2}  "
          f"{'OK' if n_beaten2 == 0 else 'NOTE'}")
