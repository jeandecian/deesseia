from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy import stats  # type: ignore[import-untyped]


class Hypothesis:
    """Hypothesis testing for statistical analysis."""

    def __init__(self) -> None:
        """Initialize the Hypothesis."""

        self._params: dict[str, Any] = {}

    def ttest_onesample(
        self,
        data: pd.Series | np.ndarray,
        popmean: float = 0,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> dict[str, Any]:
        """Perform one-sample t-test.

        Args:
            data: Sample data.
            popmean: Population mean to test against.
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater').

        Returns:
            Dictionary containing test statistic, p-value, degrees of freedom, and conclusion.
        """

        if len(data) < 2:
            raise ValueError("Sample must have at least 2 observations.")

        result = stats.ttest_1samp(data, popmean, alternative=alternative)  # type: ignore
        t_stat, p_value = cast(tuple[float, float], result)

        return {
            "statistic": float(t_stat),
            "p_value": float(p_value),
            "df": len(data) - 1,
            "alternative": alternative,
            "significant": p_value < 0.05,
            "conclusion": self._interpret_p_value(p_value),
        }

    def ttest_independent(
        self,
        group1: pd.Series | np.ndarray,
        group2: pd.Series | np.ndarray,
        equal_var: bool = True,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> dict[str, Any]:
        """Perform independent two-sample t-test.

        Args:
            group1: First sample group.
            group2: Second sample group.
            equal_var: Whether to assume equal variance (if False, uses Welch's t-test).
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater').

        Returns:
            Dictionary containing test statistic, p-value, degrees of freedom, and conclusion.
        """

        for i, group in enumerate([group1, group2]):
            if len(group) < 2:
                raise ValueError(f"Group {i+1} must have at least 2 observations.")

        result = stats.ttest_ind(  # type: ignore
            group1, group2, equal_var=equal_var, alternative=alternative
        )
        t_stat, p_value = cast(tuple[float, float], result)

        return {
            "statistic": float(t_stat),
            "p_value": float(p_value),
            "df": len(group1) + len(group2) - 2,
            "equal_var": equal_var,
            "alternative": alternative,
            "significant": p_value < 0.05,
            "conclusion": self._interpret_p_value(p_value),
        }

    def ttest_paired(
        self,
        before: pd.Series | np.ndarray,
        after: pd.Series | np.ndarray,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> dict[str, Any]:
        """Perform paired t-test (for before/after measurements).

        Args:
            before: Pre-treatment measurements.
            after: Post-treatment measurements.
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater').

        Returns:
            Dictionary containing test statistic, p-value, degrees of freedom, and conclusion.
        """

        for i, group in enumerate([before, after]):
            if len(group) < 2:
                raise ValueError(f"Group {i+1} must have at least 2 observations.")

        if len(before) != len(after):
            raise ValueError("Before and after samples must have the same length.")

        result = stats.ttest_rel(before, after, alternative=alternative)  # type: ignore
        t_stat, p_value = cast(tuple[float, float], result)

        return {
            "statistic": float(t_stat),
            "p_value": float(p_value),
            "df": len(before) - 1,
            "alternative": alternative,
            "significant": p_value < 0.05,
            "conclusion": self._interpret_p_value(p_value),
        }

    def anova_oneway(
        self,
        *groups: pd.Series | np.ndarray,
    ) -> dict[str, Any]:
        """Perform one-way ANOVA.

        Args:
            *groups: Variable number of groups to compare.

        Returns:
            Dictionary containing F-statistic, p-value, and conclusion.
        """

        if len(groups) < 2:
            raise ValueError("At least 2 groups are required for ANOVA.")

        for i, group in enumerate(groups):
            if len(group) < 2:
                raise ValueError(f"Group {i+1} must have at least 2 observations.")

        result = stats.f_oneway(*groups)  # type: ignore
        f_stat, p_value = cast(tuple[float, float], result)

        return {
            "statistic": float(f_stat),
            "p_value": float(p_value),
            "n_groups": len(groups),
            "significant": p_value < 0.05,
            "conclusion": (
                "Reject null hypothesis"
                if p_value < 0.05
                else "Fail to reject null hypothesis"
            ),
        }

    def chi_square(
        self,
        observed: pd.DataFrame | np.ndarray,
        correction: bool = True,
    ) -> dict[str, Any]:
        """Perform chi-square test for independence.

        Args:
            observed: Contingency table (observed frequencies).
            correction: Whether to apply Yates' continuity correction (for 2x2 tables).

        Returns:
            Dictionary containing chi-square statistic, p-value, degrees of freedom, and conclusion.
        """

        observed_array = np.array(observed)

        if observed_array.size == 0:
            raise ValueError("Observed table cannot be empty.")

        result = stats.chi2_contingency(observed_array, correction=correction)  # type: ignore
        chi2, p_value, dof, expected = cast(
            tuple[float, float, int, np.ndarray], result
        )

        return {
            "statistic": float(chi2),
            "p_value": float(p_value),
            "df": int(dof),
            "expected": expected.tolist(),
            "significant": p_value < 0.05,
            "conclusion": (
                "Reject null hypothesis"
                if p_value < 0.05
                else "Fail to reject null hypothesis"
            ),
        }

    def mann_whitney(
        self,
        group1: pd.Series | np.ndarray,
        group2: pd.Series | np.ndarray,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> dict[str, Any]:
        """Perform Mann-Whitney U test (non-parametric alternative to independent t-test).

        Args:
            group1: First sample group.
            group2: Second sample group.
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater').

        Returns:
            Dictionary containing U-statistic, p-value, and conclusion.
        """

        for i, group in enumerate([group1, group2]):
            if len(group) < 2:
                raise ValueError(f"Group {i+1} must have at least 2 observations.")

        result = stats.mannwhitneyu(group1, group2, alternative=alternative)  # type: ignore
        u_stat, p_value = cast(tuple[float, float], result)

        return {
            "statistic": float(u_stat),
            "p_value": float(p_value),
            "alternative": alternative,
            "significant": p_value < 0.05,
            "conclusion": self._interpret_p_value(p_value),
        }

    def wilcoxon(
        self,
        before: pd.Series | np.ndarray,
        after: pd.Series | np.ndarray,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> dict[str, Any]:
        """Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

        Args:
            before: Pre-treatment measurements.
            after: Post-treatment measurements.
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater').

        Returns:
            Dictionary containing W-statistic, p-value, and conclusion.
        """

        for i, group in enumerate([before, after]):
            if len(group) < 2:
                raise ValueError(f"Group {i+1} must have at least 2 observations.")

        if len(before) != len(after):
            raise ValueError("Before and after samples must have the same length.")

        result = stats.wilcoxon(before, after, alternative=alternative)  # type: ignore
        w_stat, p_value = cast(tuple[float, float], result)

        return {
            "statistic": float(w_stat),
            "p_value": float(p_value),
            "alternative": alternative,
            "significant": p_value < 0.05,
            "conclusion": self._interpret_p_value(p_value),
        }

    def _interpret_p_value(self, p_value: float) -> str:
        """Interpret p-value with contextual conclusion."""

        if p_value < 0.001:
            return "Strong evidence of a significant difference (p < 0.001)"
        elif p_value < 0.01:
            return "Moderate evidence of a significant difference (p < 0.01)"
        elif p_value < 0.05:
            return "Significant difference at 5% level (p < 0.05)"
        else:
            return f"No significant difference (p = {p_value:.3f})"
