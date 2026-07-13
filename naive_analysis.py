import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

def run_naive_ab_test(df, treatment_col='treatment', outcome_col='conversion', alpha=0.05):
    """
    Runs a standard naive A/B test comparing treatment vs. control conversion rates.
    Calculates lift, p-value, and confidence intervals.
    """
    print("\n--- Naive A/B Test Results ---")
    
    # Split data into treatment and control
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    # Calculate basic metrics
    n_treated = len(treated)
    n_control = len(control)
    
    conversions_treated = treated[outcome_col].sum()
    conversions_control = control[outcome_col].sum()
    
    rate_treated = conversions_treated / n_treated
    rate_control = conversions_control / n_control
    
    absolute_lift = rate_treated - rate_control
    relative_lift = (absolute_lift / rate_control) if rate_control > 0 else 0
    
    # 1. Z-Test for Proportions (p-value)
    count = np.array([conversions_treated, conversions_control])
    nobs = np.array([n_treated, n_control])
    z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
    
    # 2. Confidence Interval for the difference in proportions
    # Using 'wald' method for standard large-sample approximation
    ci_lower, ci_upper = confint_proportions_2indep(
        conversions_treated, n_treated, 
        conversions_control, n_control, 
        method='wald', alpha=alpha
    )
    
    # Format the output into a clean table format for readability
    results = pd.DataFrame({
        "Metric": [
            "Sample Size", 
            "Conversions", 
            "Conversion Rate", 
            "Absolute Lift", 
            "Relative Lift", 
            "Z-Statistic", 
            "P-Value", 
            "95% CI (Lower)", 
            "95% CI (Upper)"
        ],
        "Control": [
            f"{n_control:,}", 
            f"{int(conversions_control):,}", 
            f"{rate_control:.6f}", 
            "-", 
            "-", 
            "-", 
            "-", 
            "-", 
            "-"
        ],
        "Treatment": [
            f"{n_treated:,}", 
            f"{int(conversions_treated):,}", 
            f"{rate_treated:.6f}", 
            f"{absolute_lift:.6f}", 
            f"{relative_lift * 100:.2f}%", 
            f"{z_stat:.4f}", 
            f"{p_value:.4e}", 
            f"{ci_lower:.6f}", 
            f"{ci_upper:.6f}"
        ]
    })
    
    print(results.to_string(index=False))
    
    if p_value < alpha:
        print("\nConclusion: The difference in conversion rates is statistically significant.")
    else:
        print("\nConclusion: The difference in conversion rates is NOT statistically significant.")

    return results

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    # Mock data for testing the script (replace with actual Criteo DataFrame)
    np.random.seed(42)
    mock_treatment = np.random.binomial(n=1, p=0.85, size=1000000)
    
    # Simulate a slight lift in the treatment group
    base_rate = 0.003
    mock_outcomes = np.where(
        mock_treatment == 1, 
        np.random.binomial(n=1, p=base_rate * 1.15, size=1000000), 
        np.random.binomial(n=1, p=base_rate, size=1000000)
    )
    
    df_mock = pd.DataFrame({'treatment': mock_treatment, 'conversion': mock_outcomes})
    
    # Run the test
    run_naive_ab_test(df_mock)