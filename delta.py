"""
Delta-method asymptotic variances for regression targets in the Chain SCM.

For each acquisition subset S, the per-sample Fisher information I_S(ψ) is
combined with the gradient h = ∇_ψ g(ψ) of the scalar target g to give

    avar_S = h^T  I_S(ψ)^{-1}  h

which is the large-n limit of  n · Var(ĝ_MLE).

Prediction targets (g is a function of ψ = (σ₁², a, σ₂², b, σY²)):

    "theta"  g(ψ) = ab   h = (0, b, 0, a, 0)   proxy-only: E[Y|X1] = θ·X1
    "b"      g(ψ) = b    h = (0, 0, 0, 1, 0)   signal:     E[Y|X2] = b·X2
                                                (also optimal for X1+X2 since X1⊥Y|X2)
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from scm import ChainSCM, _parse_subset
from fisher import N_PARAMS, fisher_information


TARGETS = ["theta", "b"]
TARGET_LABELS = {
    "theta": "θ = ab  (proxy-only predictor, E[Y|X1] = θ·X1)",
    "b"    : "b       (signal predictor,     E[Y|X2] = b·X2)",
}


# ---------------------------------------------------------------------------
# Gradient vectors
# ---------------------------------------------------------------------------

def h_gradient(scm: ChainSCM, target: str) -> np.ndarray:
    """h = ∇_ψ g(ψ).

    target "theta": g(ψ) = a·b,   h = (0, b, 0, a, 0)
    target "b":     g(ψ) = b,     h = (0, 0, 0, 1, 0)
    """
    if target == "theta":
        return np.array([0.0, scm.b, 0.0, scm.a, 0.0])
    if target == "b":
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    raise ValueError(f"Unknown target {target!r}. Choose 'theta' or 'b'.")


def target_value(scm: ChainSCM, target: str) -> float:
    """True value of the target under the SCM."""
    if target == "theta": return scm.theta
    if target == "b":     return scm.b
    raise ValueError(f"Unknown target {target!r}.")


# ---------------------------------------------------------------------------
# Identifiability check and asymptotic variance
# ---------------------------------------------------------------------------

def _identifiable(h: np.ndarray, I: np.ndarray, rcond: float = 1e-8) -> bool:
    """True if h lies in the column space of I (target is estimable from S)."""
    s = np.linalg.svd(I, compute_uv=False)
    tol = rcond * s[0] if s[0] > 0 else rcond
    U, s2, Vt = np.linalg.svd(I)
    null_vecs = Vt[s2 < tol]
    if null_vecs.size == 0:
        return True
    return float(np.linalg.norm(null_vecs @ h)) < 1e-8


def avar(scm: ChainSCM, subset, target: str, rcond: float = 1e-8) -> float:
    """Per-sample asymptotic variance  h^T I_S(ψ)^+ h.

    Returns np.inf when the target is not identifiable from the subset.
    """
    I = fisher_information(scm, subset)
    h = h_gradient(scm, target)
    if not _identifiable(h, I, rcond):
        return np.inf
    return float(h @ np.linalg.pinv(I, rcond=rcond) @ h)


def avar_mixed(
    scm: ChainSCM,
    n1: float,
    n2: float,
    n12: float,
    target: str,
) -> float:
    """Asymptotic variance of target from n1 + n2 + n12 mixed observations.

    The total Fisher information adds across independent batches:
        I_total = n1·I_{X1Y} + n2·I_{X2Y} + n12·I_{X1X2Y}
    and the variance is  h^T I_total^{-1} h.
    """
    I_total = (
        n1  * fisher_information(scm, "X1Y")   +
        n2  * fisher_information(scm, "X2Y")   +
        n12 * fisher_information(scm, "X1X2Y")
    )
    h = h_gradient(scm, target)
    return float(h @ np.linalg.inv(I_total) @ h)


# ---------------------------------------------------------------------------
# Direct (efficient) estimators — used for simulation verification
# ---------------------------------------------------------------------------

def _direct_estimate(data: dict, target: str) -> float:
    """Closed-form efficient OLS estimator for target.

    theta from {X1,Y}:    θ̂ = Σ(X1·Y) / Σ(X1²)   [OLS of Y on X1]
    theta from {X1,X2,Y}: θ̂ = â·b̂ from factored MLE
    b     from {X2,Y}:    b̂ = Σ(X2·Y) / Σ(X2²)   [OLS of Y on X2]
    b     from {X1,X2,Y}: same as above
    """
    if target == "theta":
        if "X1" not in data or "Y" not in data:
            raise ValueError("theta requires X1 and Y in data")
        X1, Y = data["X1"], data["Y"]
        if "X2" in data:
            X2 = data["X2"]
            a_hat = np.dot(X1, X2) / np.dot(X1, X1)
            b_hat = np.dot(X2, Y)  / np.dot(X2, X2)
            return float(a_hat * b_hat)
        return float(np.dot(X1, Y) / np.dot(X1, X1))
    if target == "b":
        if "X2" not in data or "Y" not in data:
            raise ValueError("b requires X2 and Y in data")
        X2, Y = data["X2"], data["Y"]
        return float(np.dot(X2, Y) / np.dot(X2, X2))
    raise ValueError(f"Unknown target {target!r}")


# ---------------------------------------------------------------------------
# Cramér-Rao simulation check
# ---------------------------------------------------------------------------

def verify_avar(
    scm: ChainSCM,
    subset,
    target: str,
    n: int = 500,
    K: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Draw K datasets of n samples; compare n·Var_empirical to avar theory.

    Returns
    -------
    dict with keys: avar_theory, n_var_empirical, ratio, ok
        ok is True when ratio ∈ [0.85, 1.15], None when avar = inf.
    """
    if rng is None:
        rng = np.random.default_rng()
    subset_fs = _parse_subset(subset)
    av = avar(scm, subset_fs, target)
    if np.isinf(av):
        return {"avar_theory": np.inf, "n_var_empirical": None, "ratio": None, "ok": None}
    estimates = np.array([
        _direct_estimate(scm.sample(subset_fs, n, rng=rng), target)
        for _ in range(K)
    ])
    n_var_emp = n * float(np.var(estimates))
    ratio = n_var_emp / av
    return {
        "avar_theory"    : av,
        "n_var_empirical": n_var_emp,
        "ratio"          : ratio,
        "ok"             : 0.85 <= ratio <= 1.15,
    }


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scm = ChainSCM.from_costs(
        sigma1_sq=1.0, a=0.8, sigma2_sq=0.5, b=1.5, sigmaY_sq=0.3,
        c_X1=1.0, c_X2=3.0, c_Y=0.0,
    )

    print("=" * 72)
    print("Delta-method asymptotic variances — Chain SCM")
    print(f"ψ  = (σ₁²={scm.sigma1_sq}, a={scm.a}, σ₂²={scm.sigma2_sq}, "
          f"b={scm.b}, σY²={scm.sigmaY_sq})")
    print(f"θ  = ab = {scm.theta}")
    print("=" * 72)

    subsets_key = [
        ("X1Y",   frozenset({"X1", "Y"})),
        ("X2Y",   frozenset({"X2", "Y"})),
        ("X1X2Y", frozenset({"X1", "X2", "Y"})),
    ]

    # ── Table 1: h-vectors and per-subset asymptotic variances ───────────────
    print("\nAsymptotic variance  avar_S = h^T I_S(ψ)^{-1} h  (per sample)")
    print(f"\n{'target':>8}  {'h':>24}  {'subset':>8}  {'avar':>12}")
    print("─" * 60)
    for target in TARGETS:
        h_vec = h_gradient(scm, target)
        h_str = "(" + ", ".join(f"{v:.1f}" for v in h_vec) + ")"
        print(f"  — {TARGET_LABELS[target]}")
        for label, subset in subsets_key:
            av = avar(scm, subset, target)
            av_str = f"{av:.6f}" if not np.isinf(av) else "         ∞"
            print(f"{target:>8}  {h_str:>24}  {label:>8}  {av_str:>12}")

    # ── Table 2: Cramér-Rao verification ─────────────────────────────────────
    print("\n" + "=" * 72)
    print("Cramér-Rao verification  (K=500 trials, n=500)")
    print("Estimator: direct OLS (theta=Σ(X1Y)/Σ(X1²) or b=Σ(X2Y)/Σ(X2²))")
    print("=" * 72)

    rng = np.random.default_rng(seed=42)
    print(f"\n{'target':>8}  {'subset':>8}  {'avar_theory':>12}  "
          f"{'n·Var_emp':>10}  {'ratio':>7}  {'ok?':>5}")
    print("─" * 58)

    for target in TARGETS:
        for label, subset in subsets_key:
            res = verify_avar(scm, subset, target, n=500, K=500, rng=rng)
            av = res["avar_theory"]
            if np.isinf(av):
                print(f"{target:>8}  {label:>8}  "
                      f"{'∞':>12}  {'—':>10}  {'—':>7}  {'—':>5}")
            else:
                print(f"{target:>8}  {label:>8}  "
                      f"{av:>12.6f}  "
                      f"{res['n_var_empirical']:>10.6f}  "
                      f"{res['ratio']:>7.3f}  "
                      f"{'OK' if res['ok'] else 'FAIL':>5}")

    # ── Mixed-acquisition demo ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Mixed acquisition demo  (target: theta, n1=n2=n12=100 each)")
    print("=" * 72)
    av_mixed = avar_mixed(scm, n1=100, n2=100, n12=100, target="theta")
    av_X1Y   = avar(scm, "X1Y",   "theta")
    av_X1X2Y = avar(scm, "X1X2Y", "theta")
    print(f"\n  avar_mixed(n1=100,n2=100,n12=100)  = {av_mixed:.6f}")
    print(f"  For reference:")
    print(f"    avar_X1Y   / 300 (equiv. pure X1Y)   = {av_X1Y/300:.6f}")
    print(f"    avar_X1X2Y / 300 (equiv. pure X1X2Y) = {av_X1X2Y/300:.6f}")
