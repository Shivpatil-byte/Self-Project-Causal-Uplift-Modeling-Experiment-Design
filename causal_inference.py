import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

def check_covariate_balance(df, features, treatment_col='treatment'):
    """
    Calculates the Standardized Mean Difference (SMD) for each feature 
    to check for imbalance between treatment and control groups.
    """
    print("\n--- Covariate Balance Check (SMD) ---")
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    smd_results = {}
    for feature in features:
        mean_t = treated[feature].mean()
        mean_c = control[feature].mean()
        var_t = treated[feature].var()
        var_c = control[feature].var()
        
        # Calculate pooled standard deviation
        pooled_std = np.sqrt((var_t + var_c) / 2)
        
        # Calculate SMD
        smd = np.abs(mean_t - mean_c) / pooled_std if pooled_std > 0 else 0
        smd_results[feature] = smd
        
    smd_df = pd.DataFrame(list(smd_results.items()), columns=['Feature', 'SMD'])
    smd_df = smd_df.sort_values(by='SMD', ascending=False).reset_index(drop=True)
    
    print(smd_df.to_string(index=False))
    
    imbalanced_feats = smd_df[smd_df['SMD'] > 0.1]['Feature'].tolist()
    if imbalanced_feats:
        print(f"\n🚨 Imbalance detected (SMD > 0.1) in features: {imbalanced_feats}")
    else:
        print("\n✅ All features are well-balanced (SMD <= 0.1).")
        
    return smd_df

def propensity_score_matching(df, features, treatment_col='treatment'):
    """
    Calculates Propensity Scores using Logistic Regression and performs 
    1:1 Nearest Neighbor matching without replacement.
    """
    print("\n--- Running Propensity Score Matching ---")
    
    # 1. Estimate Propensity Scores
    X = df[features]
    y = df[treatment_col]
    
    ps_model = LogisticRegression(random_state=42, max_iter=1000)
    ps_model.fit(X, y)
    
    # Store the predicted probability of being in the treatment group
    df['propensity_score'] = ps_model.predict_proba(X)[:, 1]
    
    # 2. Nearest Neighbor Matching
    treated = df[df[treatment_col] == 1].reset_index(drop=True)
    control = df[df[treatment_col] == 0].reset_index(drop=True)
    
    # Fit KNN on the control group's propensity scores
    knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    knn.fit(control[['propensity_score']])
    
    # Find the closest control match for each treated user
    distances, indices = knn.kneighbors(treated[['propensity_score']])
    
    # Extract the matched control users
    matched_control = control.iloc[indices.flatten()]
    
    # Combine treated and matched control into a new balanced dataframe
    matched_df = pd.concat([treated, matched_control]).reset_index(drop=True)
    
    print(f"Original treated size: {len(treated):,}")
    print(f"Matched control size: {len(matched_control):,}")
    print(f"Total matched dataset size: {len(matched_df):,}")
    
    return matched_df

def estimate_causal_lift(matched_df, treatment_col='treatment', outcome_col='conversion'):
    """
    Estimates the true causal lift on the matched dataset.
    """
    print("\n--- Causal Lift Estimate (Post-Matching) ---")
    treated = matched_df[matched_df[treatment_col] == 1]
    control = matched_df[matched_df[treatment_col] == 0]
    
    rate_treated = treated[outcome_col].mean()
    rate_control = control[outcome_col].mean()
    
    causal_absolute_lift = rate_treated - rate_control
    
    print(f"Matched Control Conversion Rate:   {rate_control:.6f}")
    print(f"Matched Treatment Conversion Rate: {rate_treated:.6f}")
    print(f"Causal Absolute Lift:              {causal_absolute_lift:.6f}")
    
    return causal_absolute_lift

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    # Simulate data with a deliberate confounding variable
    np.random.seed(42)
    n_samples = 10000
    
    # Confounding feature (e.g., historical engagement level)
    f1_engagement = np.random.normal(loc=50, scale=15, size=n_samples)
    f2_random = np.random.normal(loc=10, scale=2, size=n_samples)
    
    # Higher engagement increases probability of receiving treatment
    treatment_prob = 1 / (1 + np.exp(-(f1_engagement - 50) / 10))
    treatment = np.random.binomial(1, treatment_prob)
    
    # Higher engagement AND treatment both increase conversion probability
    base_conversion_prob = 0.05 + (f1_engagement / 1000) + (treatment * 0.02)
    conversion = np.random.binomial(1, np.clip(base_conversion_prob, 0, 1))
    
    df_mock = pd.DataFrame({
        'f1_engagement': f1_engagement,
        'f2_random': f2_random,
        'treatment': treatment,
        'conversion': conversion
    })
    
    feature_cols = ['f1_engagement', 'f2_random']
    
    # 1. Check Initial Balance
    check_covariate_balance(df_mock, feature_cols)
    
    # 2. Perform Matching
    matched_data = propensity_score_matching(df_mock, feature_cols)
    
    # 3. Check Balance After Matching
    print("\n--- Post-Matching Balance Check ---")
    check_covariate_balance(matched_data, feature_cols)
    
    # 4. Estimate True Causal Lift
    estimate_causal_lift(matched_data)