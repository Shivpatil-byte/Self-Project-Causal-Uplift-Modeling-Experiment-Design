import pandas as pd
import numpy as np
from scipy.stats import chisquare
import statsmodels.stats.api as sms
from statsmodels.stats.proportion import proportion_effectsize

def load_and_inspect_criteo(file_path="criteo_uplift.csv"):
    """
    Loads the Criteo Uplift dataset and inspects the schema.
    """
    print("Loading Criteo dataset...")
    # Read only required columns to save memory if needed, or all 12 features
    df = pd.read_csv(file_path)
    
    print("\n--- Dataset Schema ---")
    print(f"Total rows: {len(df):,}")
    print(f"Features: {', '.join([col for col in df.columns if col.startswith('f')])}")
    print(f"Treatment column: 'treatment' ({df['treatment'].unique()})")
    print(f"Outcome columns: 'visit', 'conversion'")
    
    return df

def check_srm(df, treatment_col='treatment', expected_treatment_ratio=0.85):
    """
    Performs a Chi-Square Goodness-of-Fit test for Sample Ratio Mismatch (SRM).
    Criteo intended split is 85% Treatment / 15% Control.
    """
    print("\n--- Sample Ratio Mismatch (SRM) Check ---")
    
    # Observed counts
    n_total = len(df)
    n_treatment = df[treatment_col].sum()
    n_control = n_total - n_treatment
    observed = [n_treatment, n_control]
    
    # Expected counts
    expected = [
        n_total * expected_treatment_ratio, 
        n_total * (1 - expected_treatment_ratio)
    ]
    
    print(f"Observed - Treatment: {n_treatment:,} | Control: {n_control:,}")
    print(f"Expected - Treatment: {int(expected[0]):,} | Control: {int(expected[1]):,}")
    
    # Chi-Square test
    chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
    
    print(f"Chi-Square Statistic: {chi2_stat:.4f}")
    print(f"P-Value: {p_value:.4e}")
    
    # SRM Threshold is typically strict (alpha = 0.001) to avoid false positives on minor variance
    if p_value < 0.001:
        print("🚨 ALERT: Sample Ratio Mismatch (SRM) Detected! Randomization is compromised.")
    else:
        print("✅ PASS: No SRM detected. The assignment ratios match the intended design.")

def calculate_power(baseline_conversion, relative_mde, alpha=0.05, power=0.80):
    """
    Calculates the required sample size per group using statsmodels.
    """
    print("\n--- Power Analysis ---")
    
    # Calculate absolute MDE based on relative MDE
    treatment_conversion = baseline_conversion * (1 + relative_mde)
    
    # Calculate Cohen's h effect size for proportions
    effect_size = proportion_effectsize(treatment_conversion, baseline_conversion)
    
    # Initialize the Normal Independent Power instance (for Z-test of proportions)
    analysis = sms.NormalIndPower()
    
    # Solve for sample size
    required_n = analysis.solve_power(
        effect_size=effect_size, 
        alpha=alpha, 
        power=power, 
        ratio=1.0, # Assuming we want to know n for a balanced 50/50 test moving forward
        alternative='two-sided'
    )
    
    print(f"Baseline Conversion Rate: {baseline_conversion:.4f}")
    print(f"Relative MDE Target: {relative_mde*100}%")
    print(f"Absolute Conversion Target: {treatment_conversion:.4f}")
    print(f"Required Sample Size (per group): {int(np.ceil(required_n)):,}")
    print(f"Total Required Sample Size (50/50 split): {int(np.ceil(required_n) * 2):,}")

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    # 1. Run Power Analysis (assuming a historical baseline of ~0.003 for Criteo conversions)
    calculate_power(
        baseline_conversion=0.003, 
        relative_mde=0.10,  # We want to detect at least a 10% relative lift
        alpha=0.05, 
        power=0.80
    )
    
    # 2. Run SRM Check (Mock DataFrame for demonstration)
    # Replace the next 4 lines with: df = load_and_inspect_criteo("path_to_criteo.csv")
    np.random.seed(42)
    mock_treatment = np.random.binomial(n=1, p=0.85, size=13000000)
    df_mock = pd.DataFrame({'treatment': mock_treatment, 'conversion': np.zeros(13000000)})
    
    check_srm(df_mock, expected_treatment_ratio=0.85)