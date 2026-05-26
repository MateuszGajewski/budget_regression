"""
Linear Gaussian Chain SCM: X1 -> X2 -> Y

    X1 ~ N(0, sigma1_sq)
    X2 = a * X1 + eps2,   eps2 ~ N(0, sigma2_sq)
    Y  = b * X2 + epsY,   epsY ~ N(0, sigmaY_sq)

Costs are defined for every non-empty subset of {X1, X2, Y} (7 total).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, FrozenSet, Optional, Set, Union


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Subset = FrozenSet[str]
_VARS = frozenset({"X1", "X2", "Y"})

# All 7 non-empty subsets in a canonical order
ALL_SUBSETS: list[Subset] = [
    frozenset({"X1"}),
    frozenset({"X2"}),
    frozenset({"Y"}),
    frozenset({"X1", "X2"}),
    frozenset({"X1", "Y"}),
    frozenset({"X2", "Y"}),
    frozenset({"X1", "X2", "Y"}),
]


def _parse_subset(subset: Union[Subset, Set[str], str]) -> Subset:
    """Normalise a subset specification into a frozenset of variable names.

    Accepts frozenset/set of strings or a compact string such as 'X1X2Y'.
    """
    if isinstance(subset, (frozenset, set)):
        result = frozenset(subset)
    elif isinstance(subset, str):
        result = frozenset(v for v in ("X1", "X2", "Y") if v in subset)
    else:
        raise TypeError(f"Cannot parse subset from {type(subset)!r}")

    if not result:
        raise ValueError("Subset must be non-empty.")
    if not result <= _VARS:
        raise ValueError(f"Unknown variables in subset: {result - _VARS}")
    return result


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ChainSCM:
    """Linear Gaussian structural causal model for the chain X1 -> X2 -> Y.

    Parameters
    ----------
    sigma1_sq : float
        Variance of the source node X1.
    a : float
        Structural coefficient in X2 = a*X1 + eps2.
    sigma2_sq : float
        Noise variance in the X2 equation.
    b : float
        Structural coefficient in Y = b*X2 + epsY.
    sigmaY_sq : float
        Noise variance in the Y equation.
    costs : dict[frozenset[str], float]
        Per-sample observation cost for each non-empty subset of {X1, X2, Y}.
        All 7 subsets must be present; use ``make_costs`` for convenience.
    """

    def __init__(
        self,
        sigma1_sq: float,
        a: float,
        sigma2_sq: float,
        b: float,
        sigmaY_sq: float,
        costs: Dict[Subset, float],
    ) -> None:
        if sigma1_sq <= 0 or sigma2_sq <= 0 or sigmaY_sq <= 0:
            raise ValueError("All variance parameters must be positive.")

        self.sigma1_sq = float(sigma1_sq)
        self.a = float(a)
        self.sigma2_sq = float(sigma2_sq)
        self.b = float(b)
        self.sigmaY_sq = float(sigmaY_sq)

        missing = [s for s in ALL_SUBSETS if s not in costs]
        if missing:
            raise ValueError(
                f"Missing costs for subsets: {[set(s) for s in missing]}"
            )
        self.costs: Dict[Subset, float] = {
            _parse_subset(k): float(v) for k, v in costs.items()
        }

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_costs(
        cls,
        sigma1_sq: float,
        a: float,
        sigma2_sq: float,
        b: float,
        sigmaY_sq: float,
        c_X1: float,
        c_X2: float,
        c_Y: float,
        c_X1X2: Optional[float] = None,
        c_X1Y: Optional[float] = None,
        c_X2Y: Optional[float] = None,
        c_X1X2Y: Optional[float] = None,
    ) -> "ChainSCM":
        """Construct from individual subset costs.

        Joint-subset costs default to the sum of their constituent singleton
        costs when not provided explicitly.
        """
        costs = {
            frozenset({"X1"}): c_X1,
            frozenset({"X2"}): c_X2,
            frozenset({"Y"}): c_Y,
            frozenset({"X1", "X2"}): c_X1X2 if c_X1X2 is not None else c_X1 + c_X2,
            frozenset({"X1", "Y"}): c_X1Y if c_X1Y is not None else c_X1 + c_Y,
            frozenset({"X2", "Y"}): c_X2Y if c_X2Y is not None else c_X2 + c_Y,
            frozenset({"X1", "X2", "Y"}): (
                c_X1X2Y if c_X1X2Y is not None else c_X1 + c_X2 + c_Y
            ),
        }
        return cls(sigma1_sq, a, sigma2_sq, b, sigmaY_sq, costs)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def theta(self) -> float:
        """Target parameter theta = a*b (slope of E[Y | X1] = theta * X1)."""
        return self.a * self.b

    @property
    def sigma_X2_sq(self) -> float:
        """Marginal variance of X2."""
        return self.a ** 2 * self.sigma1_sq + self.sigma2_sq

    @property
    def bayes_risk_X1(self) -> float:
        """Bayes prediction risk when predicting Y from X1 only."""
        return self.b ** 2 * self.sigma2_sq + self.sigmaY_sq

    @property
    def bayes_risk_X2(self) -> float:
        """Bayes prediction risk when predicting Y from X2 (irreducible floor)."""
        return self.sigmaY_sq

    @property
    def proxy_penalty(self) -> float:
        """Extra risk from using proxy X1 instead of signal X2."""
        return self.b ** 2 * self.sigma2_sq

    def params(self) -> Dict[str, float]:
        """Return all structural parameters as a dict."""
        return {
            "sigma1_sq": self.sigma1_sq,
            "a": self.a,
            "sigma2_sq": self.sigma2_sq,
            "b": self.b,
            "sigmaY_sq": self.sigmaY_sq,
            "theta": self.theta,
            "sigma_X2_sq": self.sigma_X2_sq,
            "bayes_risk_X1": self.bayes_risk_X1,
            "bayes_risk_X2": self.bayes_risk_X2,
            "proxy_penalty": self.proxy_penalty,
        }

    # ------------------------------------------------------------------
    # Costs
    # ------------------------------------------------------------------

    def cost(self, subset: Union[Subset, Set[str], str]) -> float:
        """Per-sample observation cost for the given variable subset."""
        return self.costs[_parse_subset(subset)]

    def costs_table(self) -> pd.DataFrame:
        """Return all subset costs as a readable DataFrame."""
        rows = [
            {"subset": "{" + ", ".join(sorted(s)) + "}", "cost": self.costs[s]}
            for s in ALL_SUBSETS
        ]
        return pd.DataFrame(rows).set_index("subset")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        subset: Union[Subset, Set[str], str],
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, np.ndarray]:
        """Sample n i.i.d. observations of the variables in *subset*.

        Always generates the full causal chain (X1 -> X2 -> Y) and returns
        only the requested columns, so joint structure is preserved.

        Parameters
        ----------
        subset : frozenset | set | str
            Variables to observe, e.g. ``{'X1', 'Y'}`` or ``'X1Y'``.
        n : int
            Number of samples.
        rng : np.random.Generator, optional
            Pass for reproducibility; a fresh generator is created otherwise.

        Returns
        -------
        dict[str, np.ndarray]  – each array has shape ``(n,)``.
        """
        if rng is None:
            rng = np.random.default_rng()

        subset = _parse_subset(subset)

        X1 = rng.normal(0.0, np.sqrt(self.sigma1_sq), n)
        eps2 = rng.normal(0.0, np.sqrt(self.sigma2_sq), n)
        X2 = self.a * X1 + eps2
        epsY = rng.normal(0.0, np.sqrt(self.sigmaY_sq), n)
        Y = self.b * X2 + epsY

        full = {"X1": X1, "X2": X2, "Y": Y}
        return {v: full[v] for v in sorted(subset)}   # sorted for stable order

    def sample_df(
        self,
        subset: Union[Subset, Set[str], str],
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> pd.DataFrame:
        """Like :meth:`sample` but returns a :class:`pandas.DataFrame`."""
        return pd.DataFrame(self.sample(subset, n, rng=rng))

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ChainSCM(sigma1_sq={self.sigma1_sq}, a={self.a}, "
            f"sigma2_sq={self.sigma2_sq}, b={self.b}, "
            f"sigmaY_sq={self.sigmaY_sq}) [theta={self.theta:.4f}]"
        )


# ---------------------------------------------------------------------------
# Test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Initialise a test SCM matching the paper's setup.
    # Costs follow the additive convention from Section 2:
    #   c1 for X1, c2 for X2, c1+c2 for (X1,X2); Y is free (cost 0).
    scm = ChainSCM.from_costs(
        sigma1_sq=1.0,
        a=0.8,
        sigma2_sq=0.5,
        b=1.5,
        sigmaY_sq=0.3,
        c_X1=1.0,
        c_X2=3.0,
        c_Y=0.0,   # Y observed for free at deployment
    )

    print("=" * 60)
    print("ChainSCM: X1 -> X2 -> Y")
    print("=" * 60)
    print(scm)
    print()

    print("Structural parameters:")
    for k, v in scm.params().items():
        print(f"  {k:>16s} = {v:.4f}")
    print()

    print("Observation costs:")
    print(scm.costs_table().to_string())
    print()

    rng = np.random.default_rng(seed=42)

    # --- Sample each of the three acquisition types from the paper ---
    print("-" * 60)
    print("Sample type 1: proxy only  (X1, Y)  -- n=5")
    df1 = scm.sample_df("X1Y", n=5, rng=rng)
    print(df1.to_string(index=False))
    print()

    print("Sample type 2: signal only (X2, Y)  -- n=5")
    df2 = scm.sample_df("X2Y", n=5, rng=rng)
    print(df2.to_string(index=False))
    print()

    print("Sample type 3: paired      (X1,X2,Y) -- n=5")
    df3 = scm.sample_df("X1X2Y", n=5, rng=rng)
    print(df3.to_string(index=False))
    print()

    # --- Verify sample means / variances on a large batch ---
    print("-" * 60)
    print("Empirical check (n=100_000):")
    big = scm.sample("X1X2Y", n=100_000, rng=rng)
    print(f"  Var(X1)  expected={scm.sigma1_sq:.3f}  "
          f"got={big['X1'].var():.3f}")
    print(f"  Var(X2)  expected={scm.sigma_X2_sq:.3f}  "
          f"got={big['X2'].var():.3f}")
    print(f"  E[Y|X1 via OLS theta]  expected={scm.theta:.4f}  "
          f"got={np.polyfit(big['X1'], big['Y'], 1)[0]:.4f}")
