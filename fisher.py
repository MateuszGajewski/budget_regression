"""
Analytical Fisher Information Matrix for the ChainSCM.

For each non-empty subset S ⊆ {X1, X2, Y}, the 5×5 FIM is

    I_S(ψ)_{ij} = (1/2) tr( Σ_S⁻¹ ∂Σ_S/∂ψᵢ · Σ_S⁻¹ ∂Σ_S/∂ψⱼ )

where ψ = (sigma1_sq, a, sigma2_sq, b, sigmaY_sq) and Σ_S is the marginal
covariance of the variables in S.

Derivation sketch
-----------------
For Z_S ~ N(0, Σ_S(ψ)) the score is

    s_i(z; ψ) = −(1/2) tr(Σ_S⁻¹ dΣᵢ)  +  (1/2) z^T Σ_S⁻¹ dΣᵢ Σ_S⁻¹ z

where dΣᵢ := ∂Σ_S/∂ψᵢ.  Applying the Isserlis/Wick identity to
E[s_i · s_j], the constant × quadratic cross-terms cancel and yield the
trace formula above.

Rank expectations
-----------------
Subset        Identifiable quantities              FIM rank
{X1}          σ1²                                  1
{X2}          a²σ1²+σ2²                            1
{Y}           b²(a²σ1²+σ2²)+σY²                   1
{X1,X2}       σ1², a, σ2²                          3
{X1,Y}        σ1², ab, b²σ2²+σY²                  3
{X2,Y}        a²σ1²+σ2², b, σY²                   3
{X1,X2,Y}     all 5                                5
"""

from __future__ import annotations

from typing import FrozenSet, List, Optional, Set, Union

import numpy as np

from scm import ALL_SUBSETS, ChainSCM, _parse_subset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARAM_NAMES: List[str] = ["sigma1_sq", "a", "sigma2_sq", "b", "sigmaY_sq"]
N_PARAMS: int = len(PARAM_NAMES)
_VAR_IDX: dict[str, int] = {"X1": 0, "X2": 1, "Y": 2}

# Expected FIM rank for each subset (used in the verification summary).
EXPECTED_RANKS: dict[FrozenSet[str], int] = {
    frozenset({"X1"}): 1,
    frozenset({"X2"}): 1,
    frozenset({"Y"}): 1,
    frozenset({"X1", "X2"}): 3,
    frozenset({"X1", "Y"}): 3,
    frozenset({"X2", "Y"}): 3,
    frozenset({"X1", "X2", "Y"}): 5,
}

# ---------------------------------------------------------------------------
# Joint covariance and its Jacobians
# ---------------------------------------------------------------------------

def _full_covariance(scm: ChainSCM) -> np.ndarray:
    """Build the 3×3 joint covariance Σ_full for (X1, X2, Y).

    Variable order: row/col 0 = X1, 1 = X2, 2 = Y.

    Entries:
        Σ[0,0] = σ1²
        Σ[0,1] = a σ1²
        Σ[0,2] = ab σ1²
        Σ[1,1] = a²σ1² + σ2²      (= σ_X2²)
        Σ[1,2] = b σ_X2²
        Σ[2,2] = b² σ_X2² + σY²
    """
    s1 = scm.sigma1_sq
    a  = scm.a
    s2 = scm.sigma2_sq
    b  = scm.b
    sY = scm.sigmaY_sq
    sX2 = a**2 * s1 + s2

    return np.array([
        [s1,       a * s1,       a * b * s1        ],
        [a * s1,   sX2,          b * sX2            ],
        [a*b*s1,   b * sX2,      b**2 * sX2 + sY   ],
    ], dtype=float)


def _covariance_jacobians(scm: ChainSCM) -> List[np.ndarray]:
    """Return the five 3×3 matrices [∂Σ_full/∂ψ_0, …, ∂Σ_full/∂ψ_4].

    ψ = (sigma1_sq, a, sigma2_sq, b, sigmaY_sq)
    All five matrices are symmetric.
    """
    s1  = scm.sigma1_sq
    a   = scm.a
    s2  = scm.sigma2_sq
    b   = scm.b
    sX2 = a**2 * s1 + s2

    # ∂Σ / ∂sigma1_sq
    dS_s1 = np.array([
        [1,        a,          a * b         ],
        [a,        a**2,       a**2 * b      ],
        [a * b,    a**2 * b,   a**2 * b**2   ],
    ], dtype=float)

    # ∂Σ / ∂a
    dS_a = np.array([
        [0,          s1,           b * s1          ],
        [s1,         2*a*s1,       2*a*b*s1         ],
        [b*s1,       2*a*b*s1,     2*a*b**2*s1      ],
    ], dtype=float)

    # ∂Σ / ∂sigma2_sq
    dS_s2 = np.array([
        [0,  0,  0   ],
        [0,  1,  b   ],
        [0,  b,  b**2],
    ], dtype=float)

    # ∂Σ / ∂b
    dS_b = np.array([
        [0,       0,       a * s1       ],
        [0,       0,       sX2          ],
        [a * s1,  sX2,     2 * b * sX2  ],
    ], dtype=float)

    # ∂Σ / ∂sigmaY_sq
    dS_sY = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
    ], dtype=float)

    return [dS_s1, dS_a, dS_s2, dS_b, dS_sY]


# ---------------------------------------------------------------------------
# Marginal covariance and Jacobians for a subset
# ---------------------------------------------------------------------------

def _subset_indices(subset) -> List[int]:
    """Sorted row/column indices in Σ_full corresponding to the subset."""
    return sorted(_VAR_IDX[v] for v in _parse_subset(subset))


def marginal_covariance(scm: ChainSCM, subset) -> np.ndarray:
    """Principal submatrix of Σ_full at the rows/columns in *subset*."""
    idx = _subset_indices(subset)
    Sigma = _full_covariance(scm)
    return Sigma[np.ix_(idx, idx)]


def marginal_jacobians(scm: ChainSCM, subset) -> List[np.ndarray]:
    """List of 5 principal submatrices of ∂Σ_full/∂ψᵢ for each parameter."""
    idx = _subset_indices(subset)
    return [J[np.ix_(idx, idx)] for J in _covariance_jacobians(scm)]


# ---------------------------------------------------------------------------
# Analytical Fisher Information Matrix
# ---------------------------------------------------------------------------

def fisher_information(scm: ChainSCM, subset) -> np.ndarray:
    """Analytical 5×5 FIM for the given subset via the trace formula.

        I_S(ψ)_{ij} = (1/2) tr( Σ_S⁻¹ ∂Σ_S/∂ψᵢ · Σ_S⁻¹ ∂Σ_S/∂ψⱼ )

    The returned matrix is always 5×5.  Rows/columns for non-identifiable
    parameter directions are numerically zero (rank deficiency is expected
    for subsets that do not observe all three variables).
    """
    Sigma = marginal_covariance(scm, subset)
    Jacs  = marginal_jacobians(scm, subset)

    SigInv = np.linalg.inv(Sigma)

    # Pre-multiply: Qᵢ = Σ_S⁻¹ dΣᵢ  →  tr(Qᵢ Qⱼ) = tr(Σ⁻¹ dΣᵢ Σ⁻¹ dΣⱼ)
    Q = [SigInv @ J for J in Jacs]

    FIM = np.zeros((N_PARAMS, N_PARAMS), dtype=float)
    for i in range(N_PARAMS):
        for j in range(i, N_PARAMS):
            v = 0.5 * np.trace(Q[i] @ Q[j])
            FIM[i, j] = v
            FIM[j, i] = v

    return FIM


# ---------------------------------------------------------------------------
# MC Fisher Information Matrix (finite-difference scores — independent check)
# ---------------------------------------------------------------------------

def _scm_from_psi(psi: np.ndarray, template: ChainSCM) -> ChainSCM:
    """Construct a ChainSCM from a ψ vector, inheriting costs from *template*."""
    sigma1_sq, a, sigma2_sq, b, sigmaY_sq = psi
    return ChainSCM(sigma1_sq, a, sigma2_sq, b, sigmaY_sq, template.costs)


def fisher_information_mc(
    scm: ChainSCM,
    subset,
    n: int = 200_000,
    rng: Optional[np.random.Generator] = None,
    fd_delta: float = 1e-5,
) -> np.ndarray:
    """MC FIM estimate via finite-difference scores.

    The score is approximated by a centered finite difference that does
    NOT use any hand-derived Jacobian:

        s_i^fd(z; ψ) ≈ [log p(z; ψ+δeᵢ) − log p(z; ψ−δeᵢ)] / (2δ)

    The FIM is then estimated as the sample covariance of scores:

        Î ≈ (1/n) Σ_k s(z_k) s(z_k)^T

    This serves as a fully independent numerical check of fisher_information().

    Parameters
    ----------
    n : int
        Number of Monte Carlo samples.  200k gives ~1% relative accuracy.
    fd_delta : float
        Finite-difference step size.
    """
    if rng is None:
        rng = np.random.default_rng()

    subset = _parse_subset(subset)
    Sigma  = marginal_covariance(scm, subset)
    d      = Sigma.shape[0]

    # Draw samples from N(0, Σ_S)
    samples = rng.multivariate_normal(np.zeros(d), Sigma, size=n)  # (n, d)

    psi = np.array([scm.sigma1_sq, scm.a, scm.sigma2_sq, scm.b, scm.sigmaY_sq])

    scores = np.zeros((n, N_PARAMS), dtype=float)

    for i in range(N_PARAMS):
        # Perturbed covariance matrices
        psi_p = psi.copy(); psi_p[i] += fd_delta
        psi_m = psi.copy(); psi_m[i] -= fd_delta

        Sig_p = marginal_covariance(_scm_from_psi(psi_p, scm), subset)
        Sig_m = marginal_covariance(_scm_from_psi(psi_m, scm), subset)

        _, ld_p = np.linalg.slogdet(Sig_p)
        _, ld_m = np.linalg.slogdet(Sig_m)

        Si_p = np.linalg.inv(Sig_p)
        Si_m = np.linalg.inv(Sig_m)

        # Vectorised quadratic forms over all n samples
        qf_p = np.einsum("ni,ij,nj->n", samples, Si_p, samples)
        qf_m = np.einsum("ni,ij,nj->n", samples, Si_m, samples)

        lp_plus  = -0.5 * (ld_p + qf_p)
        lp_minus = -0.5 * (ld_m + qf_m)

        scores[:, i] = (lp_plus - lp_minus) / (2.0 * fd_delta)

    # FIM = E[s sᵀ]  (score has mean zero, so this equals the outer-product formula)
    return (scores.T @ scores) / n


# ---------------------------------------------------------------------------
# Reporting utilities
# ---------------------------------------------------------------------------

def _subset_label(subset) -> str:
    return "{" + ", ".join(sorted(_parse_subset(subset))) + "}"


def _fmt_matrix(M: np.ndarray) -> str:
    rows = []
    for row in M:
        rows.append("  " + "  ".join(f"{v:10.5f}" for v in row))
    return "\n".join(rows)


def compare_fim(
    scm: ChainSCM,
    subset,
    n: int = 200_000,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Print analytical vs MC FIM, their max absolute difference, and rank."""
    subset = _parse_subset(subset)
    label  = _subset_label(subset)

    I_an = fisher_information(scm, subset)
    I_mc = fisher_information_mc(scm, subset, n=n, rng=rng)

    rank     = np.linalg.matrix_rank(I_an, tol=1e-8)
    max_diff = np.max(np.abs(I_an - I_mc))

    print(f"\n{'─'*68}")
    print(f"Subset {label}   rank = {rank}  (expected {EXPECTED_RANKS[subset]})")
    print(f"{'─'*68}")
    print(f"ψ = {PARAM_NAMES}")
    print("Analytical FIM:")
    print(_fmt_matrix(I_an))
    print("MC FIM (finite-diff scores):")
    print(_fmt_matrix(I_mc))
    print(f"Max |analytical − MC| : {max_diff:.2e}")


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scm = ChainSCM.from_costs(
        sigma1_sq=1.0, a=0.8, sigma2_sq=0.5, b=1.5, sigmaY_sq=0.3,
        c_X1=1.0, c_X2=3.0, c_Y=0.0,
    )

    print("=" * 68)
    print("Fisher Information Matrix — ChainSCM")
    print(f"ψ = (σ1²={scm.sigma1_sq}, a={scm.a}, σ2²={scm.sigma2_sq}, "
          f"b={scm.b}, σY²={scm.sigmaY_sq})")
    print(f"θ = ab = {scm.theta:.4f}")
    print("=" * 68)

    rng = np.random.default_rng(seed=0)
    for subset in ALL_SUBSETS:
        compare_fim(scm, subset, n=2_000_000, rng=rng)

    # ── Rank summary ──────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("Rank summary")
    print(f"{'Subset':<22} {'Expected':>8}  {'Got':>5}  {'Match':>6}")
    print("─" * 50)
    for subset in ALL_SUBSETS:
        label = _subset_label(subset)
        I     = fisher_information(scm, subset)
        got   = np.linalg.matrix_rank(I, tol=1e-8)
        exp   = EXPECTED_RANKS[subset]
        mark  = "OK" if got == exp else "FAIL"
        print(f"{label:<22} {exp:>8}  {got:>5}  {mark:>6}")
