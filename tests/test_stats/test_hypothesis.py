from typing import Any

import numpy as np
import pytest

from deesseia.stats.hypothesis import Hypothesis


class TestHypothesis:
    """Test hypothesis testing functionality."""

    def test_ttest_onesample_default(self, sample_data_ttest: np.ndarray) -> None:
        """Test one-sample t-test with default parameters."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.ttest_onesample(sample_data_ttest)

        assert "statistic" in result
        assert "p_value" in result
        assert "df" in result
        assert "alternative" in result
        assert "significant" in result
        assert "conclusion" in result
        assert result["df"] == 9

    def test_ttest_onesample_with_popmean(self, sample_data_ttest: np.ndarray) -> None:
        """Test one-sample t-test with custom population mean."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.ttest_onesample(sample_data_ttest, popmean=5)

        assert "statistic" in result
        assert "p_value" in result

    def test_ttest_onesample_less(self, sample_data_ttest: np.ndarray) -> None:
        """Test one-sample t-test with 'less' alternative."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.ttest_onesample(
            sample_data_ttest, popmean=5, alternative="less"
        )

        assert result["alternative"] == "less"

    def test_ttest_onesample_greater(self, sample_data_ttest: np.ndarray) -> None:
        """Test one-sample t-test with 'greater' alternative."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.ttest_onesample(
            sample_data_ttest, popmean=5, alternative="greater"
        )

        assert result["alternative"] == "greater"

    def test_ttest_onesample_insufficient_data(
        self, sample_data_single_observation: np.ndarray
    ) -> None:
        """Test one-sample t-test with insufficient data."""

        hyp: Hypothesis = Hypothesis()

        with pytest.raises(
            ValueError, match="Sample must have at least 2 observations"
        ):
            hyp.ttest_onesample(sample_data_single_observation)

    def test_ttest_independent_default(
        self,
        sample_data_group1: np.ndarray,
        sample_data_group2: np.ndarray,
    ) -> None:
        """Test independent t-test with default parameters."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.ttest_independent(
            sample_data_group1, sample_data_group2
        )

        assert "statistic" in result
        assert "p_value" in result
        assert "df" in result
        assert result["equal_var"] is True

    def test_ttest_independent_welch(
        self,
        sample_data_group1: np.ndarray,
        sample_data_group2_welch: np.ndarray,
    ) -> None:
        """Test independent t-test with Welch's correction (equal_var=False)."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.ttest_independent(
            sample_data_group1, sample_data_group2_welch, equal_var=False
        )

        assert result["equal_var"] is False

    def test_ttest_independent_less(
        self,
        sample_data_group1: np.ndarray,
        sample_data_group2: np.ndarray,
    ) -> None:
        """Test independent t-test with 'less' alternative."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.ttest_independent(
            sample_data_group1, sample_data_group2, alternative="less"
        )

        assert result["alternative"] == "less"

    def test_ttest_independent_insufficient_data(
        self,
        sample_data_single_observation: np.ndarray,
        sample_data_two_observations: np.ndarray,
    ) -> None:
        """Test independent t-test with insufficient data."""

        hyp: Hypothesis = Hypothesis()

        with pytest.raises(
            ValueError, match="Group 1 must have at least 2 observations"
        ):
            hyp.ttest_independent(
                sample_data_single_observation, sample_data_two_observations
            )

    def test_ttest_paired_default(
        self,
        sample_data_ttest: np.ndarray,
        sample_data_paired_after: np.ndarray,
    ) -> None:
        """Test paired t-test with default parameters."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.ttest_paired(
            sample_data_ttest, sample_data_paired_after
        )

        assert "statistic" in result
        assert "p_value" in result
        assert "df" in result
        assert result["df"] == 9

    def test_ttest_paired_less(
        self,
        sample_data_ttest: np.ndarray,
        sample_data_paired_after: np.ndarray,
    ) -> None:
        """Test paired t-test with 'less' alternative."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.ttest_paired(
            sample_data_ttest, sample_data_paired_after, alternative="less"
        )

        assert result["alternative"] == "less"

    def test_ttest_paired_insufficient_data(
        self,
        sample_data_single_observation: np.ndarray,
        sample_data_two_observations: np.ndarray,
    ) -> None:
        """Test paired t-test with insufficient data."""

        hyp: Hypothesis = Hypothesis()

        with pytest.raises(
            ValueError, match="Group 1 must have at least 2 observations"
        ):
            hyp.ttest_paired(
                sample_data_single_observation, sample_data_two_observations
            )

    def test_ttest_paired_mismatched_length(
        self,
        sample_data_three_observations: np.ndarray,
        sample_data_two_observations_mismatch: np.ndarray,
    ) -> None:
        """Test paired t-test with mismatched lengths."""

        hyp: Hypothesis = Hypothesis()

        with pytest.raises(
            ValueError, match="Before and after samples must have the same length"
        ):
            hyp.ttest_paired(
                sample_data_three_observations, sample_data_two_observations_mismatch
            )

    def test_anova_oneway_default(
        self,
        sample_data_group1: np.ndarray,
        sample_data_group2: np.ndarray,
        sample_data_group3: np.ndarray,
    ) -> None:
        """Test one-way ANOVA with default parameters."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.anova_oneway(
            sample_data_group1, sample_data_group2, sample_data_group3
        )

        assert "statistic" in result
        assert "p_value" in result
        assert "n_groups" in result
        assert result["n_groups"] == 3

    def test_anova_oneway_two_groups(
        self,
        sample_data_group1: np.ndarray,
        sample_data_group2: np.ndarray,
    ) -> None:
        """Test one-way ANOVA with two groups."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.anova_oneway(
            sample_data_group1, sample_data_group2
        )

        assert result["n_groups"] == 2

    def test_anova_oneway_insufficient_groups(
        self, sample_data_group1: np.ndarray
    ) -> None:
        """Test one-way ANOVA with insufficient groups."""

        hyp: Hypothesis = Hypothesis()

        with pytest.raises(
            ValueError, match="At least 2 groups are required for ANOVA"
        ):
            hyp.anova_oneway(sample_data_group1)

    def test_anova_oneway_insufficient_data(
        self,
        sample_data_single_observation: np.ndarray,
        sample_data_group2: np.ndarray,
    ) -> None:
        """Test one-way ANOVA with insufficient data in one group."""

        hyp: Hypothesis = Hypothesis()

        with pytest.raises(
            ValueError, match="Group 1 must have at least 2 observations"
        ):
            hyp.anova_oneway(sample_data_single_observation, sample_data_group2)

    def test_chi_square_default(self, sample_data_chi_square: np.ndarray) -> None:
        """Test chi-square test with default parameters."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.chi_square(sample_data_chi_square)

        assert "statistic" in result
        assert "p_value" in result
        assert "df" in result
        assert "expected" in result
        assert result["df"] == 1

    def test_chi_square_no_correction(self, sample_data_chi_square: np.ndarray) -> None:
        """Test chi-square test without Yates' correction."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.chi_square(
            sample_data_chi_square, correction=False
        )

        assert "statistic" in result

    def test_chi_square_empty_table(self) -> None:
        """Test chi-square test with empty table."""

        observed: np.ndarray = np.array([])
        hyp: Hypothesis = Hypothesis()

        with pytest.raises(ValueError, match="Observed table cannot be empty"):
            hyp.chi_square(observed)

    def test_mann_whitney_default(
        self,
        sample_data_group1: np.ndarray,
        sample_data_group2: np.ndarray,
    ) -> None:
        """Test Mann-Whitney U test with default parameters."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.mann_whitney(
            sample_data_group1, sample_data_group2
        )

        assert "statistic" in result
        assert "p_value" in result
        assert "alternative" in result

    def test_mann_whitney_less(
        self,
        sample_data_group1: np.ndarray,
        sample_data_group2: np.ndarray,
    ) -> None:
        """Test Mann-Whitney U test with 'less' alternative."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.mann_whitney(
            sample_data_group1, sample_data_group2, alternative="less"
        )

        assert result["alternative"] == "less"

    def test_mann_whitney_insufficient_data(
        self,
        sample_data_single_observation: np.ndarray,
        sample_data_two_observations: np.ndarray,
    ) -> None:
        """Test Mann-Whitney U test with insufficient data."""

        hyp: Hypothesis = Hypothesis()

        with pytest.raises(
            ValueError, match="Group 1 must have at least 2 observations"
        ):
            hyp.mann_whitney(
                sample_data_single_observation, sample_data_two_observations
            )

    def test_wilcoxon_default(
        self,
        sample_data_ttest: np.ndarray,
        sample_data_paired_after: np.ndarray,
    ) -> None:
        """Test Wilcoxon signed-rank test with default parameters."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.wilcoxon(
            sample_data_ttest, sample_data_paired_after
        )

        assert "statistic" in result
        assert "p_value" in result
        assert "alternative" in result

    def test_wilcoxon_less(
        self,
        sample_data_ttest: np.ndarray,
        sample_data_paired_after: np.ndarray,
    ) -> None:
        """Test Wilcoxon signed-rank test with 'less' alternative."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.wilcoxon(
            sample_data_ttest, sample_data_paired_after, alternative="less"
        )

        assert result["alternative"] == "less"

    def test_wilcoxon_insufficient_data(
        self,
        sample_data_single_observation: np.ndarray,
        sample_data_two_observations: np.ndarray,
    ) -> None:
        """Test Wilcoxon signed-rank test with insufficient data."""

        hyp: Hypothesis = Hypothesis()

        with pytest.raises(
            ValueError, match="Group 1 must have at least 2 observations"
        ):
            hyp.wilcoxon(sample_data_single_observation, sample_data_two_observations)

    def test_wilcoxon_mismatched_length(
        self,
        sample_data_three_observations: np.ndarray,
        sample_data_two_observations_mismatch: np.ndarray,
    ) -> None:
        """Test Wilcoxon signed-rank test with mismatched lengths."""

        hyp: Hypothesis = Hypothesis()

        with pytest.raises(
            ValueError, match="Before and after samples must have the same length"
        ):
            hyp.wilcoxon(
                sample_data_three_observations, sample_data_two_observations_mismatch
            )

    # ============================================================
    # Tests for p-value interpretation through public methods
    # ============================================================

    def test_ttest_onesample_conclusion_p_less_than_001(
        self, sample_data_very_strong_evidence: np.ndarray
    ) -> None:
        """Test that t-test returns correct conclusion for p < 0.001."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.ttest_onesample(
            sample_data_very_strong_evidence, popmean=0
        )

        assert "conclusion" in result
        assert result["p_value"] < 0.001
        assert (
            result["conclusion"]
            == "Strong evidence of a significant difference (p < 0.001)"
        )

    def test_ttest_onesample_conclusion_moderate_evidence(
        self, sample_data_moderate_evidence: np.ndarray
    ) -> None:
        """Test that t-test returns correct conclusion for 0.001 <= p < 0.01."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.ttest_onesample(
            sample_data_moderate_evidence, popmean=0
        )

        assert "conclusion" in result
        assert result["p_value"] < 0.01
        assert (
            result["conclusion"]
            == "Moderate evidence of a significant difference (p < 0.01)"
        )

    def test_ttest_onesample_conclusion_significant_at_5_percent(
        self, sample_data_strong_evidence: np.ndarray
    ) -> None:
        """Test that t-test returns correct conclusion for 0.01 <= p < 0.05."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.ttest_onesample(
            sample_data_strong_evidence, popmean=0
        )

        assert "conclusion" in result
        assert 0.01 <= result["p_value"] < 0.05
        assert result["conclusion"] == "Significant difference at 5% level (p < 0.05)"

    def test_ttest_onesample_conclusion_no_significant_difference(
        self, sample_data_no_evidence: np.ndarray
    ) -> None:
        """Test that t-test returns correct conclusion for p >= 0.05."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.ttest_onesample(sample_data_no_evidence, popmean=0)

        assert "conclusion" in result
        assert result["p_value"] >= 0.05
        assert "No significant difference" in result["conclusion"]

    def test_independent_ttest_conclusion_p_less_than_001(
        self,
        sample_group1_very_strong_evidence: np.ndarray,
        sample_group2_very_strong_evidence: np.ndarray,
    ) -> None:
        """Test that independent t-test returns correct conclusion for p < 0.001."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.ttest_independent(
            sample_group1_very_strong_evidence, sample_group2_very_strong_evidence
        )

        assert "conclusion" in result
        assert result["p_value"] < 0.001
        assert (
            result["conclusion"]
            == "Strong evidence of a significant difference (p < 0.001)"
        )

    def test_chi_square_conclusion_reject_null(
        self, sample_chi_square_strong_evidence: np.ndarray
    ) -> None:
        """Test that chi-square returns 'Reject null hypothesis' for p < 0.05."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.chi_square(sample_chi_square_strong_evidence)

        assert "conclusion" in result
        assert result["p_value"] < 0.001
        assert result["conclusion"] == "Reject null hypothesis"

    def test_mann_whitney_conclusion_no_significant_difference(
        self,
        sample_group1_mann_whitney_no_evidence: np.ndarray,
        sample_group2_mann_whitney_no_evidence: np.ndarray,
    ) -> None:
        """Test that Mann-Whitney returns a conclusion for p >= 0.05."""

        hyp: Hypothesis = Hypothesis()
        result: dict[str, Any] = hyp.mann_whitney(
            sample_group1_mann_whitney_no_evidence,
            sample_group2_mann_whitney_no_evidence,
        )

        assert "conclusion" in result
        assert "No significant difference" in result["conclusion"]
