"""
Maximum Likelihood Estimation for the Chain SCM.

Given samples of possibly different types (e.g. n1 pairs (X1,Y), n2 pairs
(X2,Y), n12 triplets (X1,X2,Y)), fit the parameter vector
ψ = (σ₁², a, σ₂², b, σY²) by maximising the joint log-likelihood

    ℓ(ψ) = Σ_k  −(n_k/2)[log|Σ_{S_k}(ψ)| + tr(Σ_{S_k}(ψ)⁻¹ Ŝ_k)]

where Ŝ_k = (1/n_k) Z_k^T Z_k is the sample second-moment matrix for batch k.

Optimisation is performed in the unconstrained reparameterisation
    τ = (log σ₁², a, log σ₂², b, log σY²)
so variance parameters are automatically positive.
"""

from __future__ import annotations

import warnings
from typing import Dict, FrozenSet, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize

from scm import ChainSCM, _parse_subset
from fisher import (
    PARAM_NAMES,
    N_PARAMS,
    marginal_covariance,
    _full_covariance,
    fisher_information,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_data_matrix(
    sample: Union[np.ndarray, Dict[str, np.ndarray]],
) -> np.ndarray:
    """Convert scm.sample() output or raw array to an (n, d) data matrix.

    Columns follow canonical alphabetical order: X1 → col 0, X2 → col 1,
    Y → col 2.  For partial subsets the same ordering applies to the
    variables present (e.g. {X1, Y} → [X1, Y] in that order).
    """
    if isinstance(sample, dict):
        return np.column_stack([sample[v] for v in sorted(sample)])
    return np.asarray(sample, dtype=float)


def _psi_from_scm(scm: ChainSCM) -> np.ndarray:
    return np.array([scm.sigma1_sq, scm.a, scm.sigma2_sq, scm.b, scm.sigmaY_sq])


def _scm_from_psi(psi: np.ndarray, template: ChainSCM) -> ChainSCM:
    return ChainSCM(
        float(psi[0]), float(psi[1]), float(psi[2]),
        float(psi[3]), float(psi[4]),
        template.costs,
    )


def _tau_to_psi(tau: np.ndarray) -> np.ndarray:
    """Unconstrained τ → ψ.  Variance positions are exp-transformed."""
    return np.array([
        np.exp(tau[0]),  # σ₁²
        tau[1],          # a
        np.exp(tau[2]),  # σ₂²
        tau[3],          # b
        np.exp(tau[4]),  # σY²
    ])


def _psi_to_tau(psi: np.ndarray) -> np.ndarray:
    """ψ → unconstrained τ (inverse of _tau_to_psi)."""
    return np.array([
        np.log(psi[0]),
        psi[1],
        np.log(psi[2]),
        psi[3],
        np.log(psi[4]),
    ])


# ---------------------------------------------------------------------------
# Sample statistics
# ---------------------------------------------------------------------------

def sample_second_moment(
    data_dict: Dict,
) -> Dict[FrozenSet[str], Tuple[int, np.ndarray]]:
    """Compute the sample-size and second-moment matrix for each data batch.

    Parameters
    ----------
    data_dict : dict
        Maps each subset specification to its data.  Values can be:
        - np.ndarray of shape (n, d) — columns in sorted variable order
        - dict {var_name: np.ndarray of shape (n,)} — from scm.sample()

    Returns
    -------
    dict  frozenset[str] → (n_k, Ŝ_k)
        n_k : number of observations in batch k
        Ŝ_k : (1/n_k) Z^T Z  of shape (d_k, d_k)
    """
    result: Dict[FrozenSet[str], Tuple[int, np.ndarray]] = {}
    for key, data in data_dict.items():
        subset = _parse_subset(key)
        mat = _to_data_matrix(data)
        n = mat.shape[0]
        result[subset] = (n, (mat.T @ mat) / n)
    return result


# ---------------------------------------------------------------------------
# Data-driven warm start
# ---------------------------------------------------------------------------

def _init_tau_from_moments(
    moments: Dict[FrozenSet[str], Tuple[int, np.ndarray]],
) -> np.ndarray:
    """Derive an initial τ from sample second moments — no outside knowledge.

    Column order inside each Ŝ_k follows sorted variable names:
        {X1,X2,Y} → [0=X1, 1=X2, 2=Y]
        {X1,X2}   → [0=X1, 1=X2]
        {X2,Y}    → [0=X2, 1=Y]
        {X1,Y}    → [0=X1, 1=Y]

    Priority: use the richest available subset first.  Defaults (1.0 / 0.5)
    are used for parameters that no available subset identifies.
    """
    sigma1_sq = 1.0
    a         = 0.5
    sigma2_sq = 1.0
    b         = 0.5
    sigmaY_sq = 1.0

    def _S(vars_):
        k = frozenset(vars_)
        return moments[k][1] if k in moments else None

    S_full = _S({"X1", "X2", "Y"})
    if S_full is not None:
        # Closed-form OLS estimates from the full joint second moment.
        sigma1_sq = float(S_full[0, 0])
        a         = float(S_full[0, 1] / sigma1_sq)    if sigma1_sq > 1e-9 else 0.5
        sX2       = float(S_full[1, 1])
        sigma2_sq = float(sX2 - a**2 * sigma1_sq)
        b         = float(S_full[1, 2] / sX2)          if sX2 > 1e-9 else 0.5
        sigmaY_sq = float(S_full[2, 2] - b**2 * sX2)
    else:
        # Build estimates piecemeal from whatever subsets are available.

        # X1X2 or X1-only → σ₁², a, σ₂²
        S12 = _S({"X1", "X2"})
        S1  = _S({"X1"})
        if S12 is not None:
            sigma1_sq = float(S12[0, 0])
            a         = float(S12[0, 1] / sigma1_sq) if sigma1_sq > 1e-9 else 0.5
            sX2       = float(S12[1, 1])
            sigma2_sq = float(sX2 - a**2 * sigma1_sq)
        elif S1 is not None:
            sigma1_sq = float(S1[0, 0])

        # X2Y → b, σY²
        S2y = _S({"X2", "Y"})
        if S2y is not None:
            sX2_obs   = float(S2y[0, 0])
            b         = float(S2y[0, 1] / sX2_obs) if sX2_obs > 1e-9 else 0.5
            sigmaY_sq = float(S2y[1, 1] - b**2 * sX2_obs)
            if S12 is None:
                # σ_X2² is observed but a, σ₁², σ₂² are not separately identified;
                # keep a=0.5 default and set σ₂² = σ_X2² − a²σ₁²
                sigma2_sq = sX2_obs - a**2 * sigma1_sq

        # X1Y → σ₁² (and ab product if b is still unknown)
        S1y = _S({"X1", "Y"})
        if S1y is not None:
            if S12 is None and S1 is None:
                sigma1_sq = float(S1y[0, 0])
            if S2y is None:
                # Estimate ab from the cross moment, split with current a
                ab_hat = (float(S1y[0, 1] / S1y[0, 0])
                          if S1y[0, 0] > 1e-9 else 0.25)
                b = ab_hat / a if abs(a) > 1e-3 else ab_hat

    # Clip to valid range
    sigma1_sq = max(sigma1_sq, 1e-3)
    sigma2_sq = max(sigma2_sq, 1e-3)
    sigmaY_sq = max(sigmaY_sq, 1e-3)

    return _psi_to_tau(np.array([sigma1_sq, a, sigma2_sq, b, sigmaY_sq]))


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def neg_log_likelihood(
    tau: np.ndarray,
    template: ChainSCM,
    moments: Dict[FrozenSet[str], Tuple[int, np.ndarray]],
) -> float:
    """
    Negative log-likelihood (up to additive constants) at unconstrained τ.

        −ℓ(ψ(τ)) = Σ_k (n_k/2) [log|Σ_{S_k}(ψ)| + tr(Σ_{S_k}(ψ)⁻¹ Ŝ_k)]

    Returns a large sentinel value (1e10) for numerically invalid parameters.
    """
    try:
        psi = _tau_to_psi(tau)
        scm = _scm_from_psi(psi, template)
    except (ValueError, OverflowError):
        return 1e10

    total = 0.0
    for subset, (n_k, S_hat) in moments.items():
        try:
            Sigma = marginal_covariance(scm, subset)
            sign, logdet = np.linalg.slogdet(Sigma)
            if sign <= 0:
                return 1e10
            SigInv = np.linalg.inv(Sigma)
            total += (n_k / 2.0) * (logdet + np.trace(SigInv @ S_hat))
        except np.linalg.LinAlgError:
            return 1e10

    return float(total)


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------

def mle_fit(
    template: ChainSCM,
    data_dict: Dict,
    init_tau: Optional[np.ndarray] = None,
    method: str = "L-BFGS-B",
    n_restarts: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> ChainSCM:
    """
    Fit the chain SCM to data by maximum likelihood.

    Supports mixed acquisition types: data_dict may contain entries for
    multiple subsets simultaneously; their log-likelihoods are summed.

    Parameters
    ----------
    template : ChainSCM
        Provides the cost structure carried into the returned SCM.
        Its ψ values are NOT used; the warm start is derived from data.
    data_dict : dict
        Maps subset spec → data (ndarray or {var_name: array}).
    init_tau : array of shape (5,), optional
        Initial unconstrained τ.  Defaults to a data-driven estimate
        via _init_tau_from_moments (no outside knowledge required).
    method : str
        scipy.optimize.minimize method.  'L-BFGS-B' uses numerical
        gradients and handles box-constraint-free smooth objectives well.
    n_restarts : int
        Number of optimisation runs; the best (lowest −ℓ) result is
        returned.  The first run uses init_tau; subsequent runs use
        randomly perturbed starts.
    rng : np.random.Generator, optional

    Returns
    -------
    ChainSCM  with ψ̂ and template's costs.
    """
    if rng is None:
        rng = np.random.default_rng()

    moments = sample_second_moment(data_dict)

    if init_tau is None:
        init_tau = _init_tau_from_moments(moments)

    starts = [init_tau] + [
        init_tau + rng.standard_normal(N_PARAMS) * 0.5
        for _ in range(n_restarts - 1)
    ]

    best_val = np.inf
    best_tau = init_tau.copy()

    for tau0 in starts:
        res = minimize(
            neg_log_likelihood,
            tau0,
            args=(template, moments),
            method=method,
            options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-9},
        )
        if res.fun < best_val:
            best_val = res.fun
            best_tau = res.x

    if best_val >= 1e9:
        warnings.warn("mle_fit: optimisation did not reach a valid solution.")

    return _scm_from_psi(_tau_to_psi(best_tau), template)


# ---------------------------------------------------------------------------
# Closed-form MLE for complete (X1, X2, Y) observations
# ---------------------------------------------------------------------------

def closed_form_mle(
    data: Union[np.ndarray, Dict[str, np.ndarray]],
) -> Dict[str, float]:
    """
    Exact MLE for complete (X1, X2, Y) observations.

    The SCM factorises as p(X1) · p(X2|X1) · p(Y|X2), yielding three
    independent OLS problems (no centering; model has zero mean):

        σ̂₁²  = mean(X1²)
        â     = mean(X1·X2) / mean(X1²)
        σ̂₂²  = mean((X2 − â X1)²)
        b̂     = mean(X2·Y)  / mean(X2²)
        σ̂Y²  = mean((Y  − b̂ X2)²)

    Parameters
    ----------
    data : np.ndarray of shape (n, 3) with columns [X1, X2, Y],
           or dict {"X1": ..., "X2": ..., "Y": ...} (output of scm.sample()).

    Returns
    -------
    dict with keys == PARAM_NAMES, in the same order.
    """
    mat = _to_data_matrix(data)
    X1, X2, Y = mat[:, 0], mat[:, 1], mat[:, 2]

    sigma1_sq = float(np.mean(X1 ** 2))
    a_hat     = float(np.mean(X1 * X2) / sigma1_sq)
    sigma2_sq = float(np.mean((X2 - a_hat * X1) ** 2))
    sX2_sq    = float(np.mean(X2 ** 2))
    b_hat     = float(np.mean(X2 * Y) / sX2_sq)
    sigmaY_sq = float(np.mean((Y - b_hat * X2) ** 2))

    return {
        "sigma1_sq": sigma1_sq,
        "a":         a_hat,
        "sigma2_sq": sigma2_sq,
        "b":         b_hat,
        "sigmaY_sq": sigmaY_sq,
    }


# ---------------------------------------------------------------------------
# __main__: three verification checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    true_scm = ChainSCM.from_costs(
        sigma1_sq=1.0, a=0.8, sigma2_sq=0.5, b=1.5, sigmaY_sq=0.3,
        c_X1=1.0, c_X2=3.0, c_Y=0.0,
    )
    psi_true = _psi_from_scm(true_scm)
    rng = np.random.default_rng(seed=0)

    # ==================================================================
    # CHECK 1 — Optimizer ≈ closed-form MLE  (full data, n=2000)
    # ==================================================================
    print("=" * 68)
    print("CHECK 1 — Optimizer vs closed-form MLE  (n=2000, full data)")
    print("=" * 68)

    data_full = true_scm.sample("X1X2Y", n=2000, rng=rng)
    scm_opt   = mle_fit(true_scm, {"X1X2Y": data_full}, rng=rng)
    psi_cf    = closed_form_mle(data_full)

    print(f"\n{'param':>12}  {'true':>8}  {'closed-form':>12}  {'optimizer':>10}  {'|opt−cf|':>9}")
    print("─" * 60)
    for name, tv in zip(PARAM_NAMES, psi_true):
        cf  = psi_cf[name]
        opt = getattr(scm_opt, name)
        print(f"{name:>12}  {tv:>8.4f}  {cf:>12.6f}  {opt:>10.6f}  {abs(opt - cf):>9.2e}")

    max_diff = max(abs(getattr(scm_opt, p) - psi_cf[p]) for p in PARAM_NAMES)
    print(f"\nMax |optimizer − closed-form|: {max_diff:.2e}  "
          f"{'OK' if max_diff < 1e-4 else 'FAIL'}")

    # ==================================================================
    # CHECK 2 — Large-n consistency: marginal covariance converges
    #           (test each of the three main acquisition subsets)
    # ==================================================================
    print("\n" + "=" * 68)
    print("CHECK 2 — Large-n consistency  (n=100 000, warm-started from true ψ)")
    print("=" * 68)

    for s_str in ("X1Y", "X2Y", "X1X2Y"):
        subset      = _parse_subset(s_str)
        label       = "{" + ", ".join(sorted(subset)) + "}"
        data_big    = true_scm.sample(s_str, n=100_000, rng=rng)
        scm_fit     = mle_fit(true_scm, {s_str: data_big}, n_restarts=1, rng=rng)
        Sigma_fit   = marginal_covariance(scm_fit,  subset)
        Sigma_truth = marginal_covariance(true_scm, subset)
        err         = np.linalg.norm(Sigma_fit - Sigma_truth, "fro")
        ok          = err < 0.03

        print(f"\nSubset {label}")
        print(f"  True     Σ_S:  {np.array2string(Sigma_truth, precision=5, suppress_small=True)}")
        print(f"  Fitted   Σ_S:  {np.array2string(Sigma_fit,   precision=5, suppress_small=True)}")
        print(f"  ‖Σ_S(ψ̂) − Σ_S(ψ*)‖_F = {err:.2e}  {'OK' if ok else 'FAIL'}")

    # ==================================================================
    # CHECK 3 — Cramér-Rao bound via closed-form MLE
    #           (K=500 independent fits, n=500, full data)
    # ==================================================================
    print("\n" + "=" * 68)
    print("CHECK 3 — Cramér-Rao bound  (K=500 fits, n=500, full data)")
    print("=" * 68)

    K, n3 = 500, 500
    I_full = fisher_information(true_scm, "X1X2Y")
    crb    = np.diag(np.linalg.inv(I_full))   # [I^{-1}]_{ii}

    # Use closed_form_mle (exact + fast) for the 500 repetitions
    psi_ests = np.array([
        [psi_cf[p]
         for p in PARAM_NAMES
         for psi_cf in [closed_form_mle(true_scm.sample("X1X2Y", n=n3, rng=rng))]]
        for _ in range(K)
    ])   # shape (K, 5)

    emp_n_var = n3 * psi_ests.var(axis=0)

    print(f"\n{'param':>12}  {'CRB':>9}  {'n·Var_emp':>10}  {'ratio':>7}  {'ok?':>5}")
    print("─" * 52)
    for i, name in enumerate(PARAM_NAMES):
        ratio = emp_n_var[i] / crb[i]
        ok    = 0.85 <= ratio <= 1.15
        print(f"{name:>12}  {crb[i]:>9.5f}  {emp_n_var[i]:>10.5f}  "
              f"{ratio:>7.3f}  {'OK' if ok else 'FAIL'}")
