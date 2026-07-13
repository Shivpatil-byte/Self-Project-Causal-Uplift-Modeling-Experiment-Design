import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

def train_t_learner(df, features, treatment_col='treatment', outcome_col='conversion'):
    """
    Trains a T-Learner using two separate XGBoost classifiers.
    """
    print("\n--- Training T-Learner (Uplift Model) ---")
    
    # Split data by treatment status
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    # Initialize two separate XGBoost models
    model_t = xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
    model_c = xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
    
    # Fit the models
    print("Training Treatment Model...")
    model_t.fit(treated[features], treated[outcome_col])
    
    print("Training Control Model...")
    model_c.fit(control[features], control[outcome_col])
    
    # Predict probabilities for ALL users using BOTH models
    # We want the probability of the positive class (conversion = 1)
    df_results = df.copy()
    df_results['prob_treat'] = model_t.predict_proba(df[features])[:, 1]
    df_results['prob_control'] = model_c.predict_proba(df[features])[:, 1]
    
    # Calculate Individual Uplift
    df_results['predicted_uplift'] = df_results['prob_treat'] - df_results['prob_control']
    
    return df_results, model_t, model_c

def segment_users(df_results):
    """
    Buckets users into quartiles based on their predicted uplift.
    """
    print("\n--- User Segmentation by Uplift ---")
    
    # Create 4 equal-sized buckets based on predicted uplift
    df_results['uplift_quartile'] = pd.qcut(df_results['predicted_uplift'], q=4, labels=['Q4 (Lowest)', 'Q3', 'Q2', 'Q1 (Highest)'])
    
    # Calculate the actual conversion rates for treatment and control within each predicted bucket
    segment_metrics = []
    for quartile in ['Q1 (Highest)', 'Q2', 'Q3', 'Q4 (Lowest)']:
        subset = df_results[df_results['uplift_quartile'] == quartile]
        
        t_conv = subset[subset['treatment'] == 1]['conversion'].mean()
        c_conv = subset[subset['treatment'] == 0]['conversion'].mean()
        actual_lift = t_conv - c_conv
        
        segment_metrics.append({
            'Segment': quartile,
            'Avg Predicted Uplift': subset['predicted_uplift'].mean(),
            'Actual Treatment Conv Rate': t_conv,
            'Actual Control Conv Rate': c_conv,
            'Actual Absolute Lift': actual_lift,
            'User Count': len(subset)
        })
        
    segment_df = pd.DataFrame(segment_metrics)
    print(segment_df.to_string(index=False))
    return segment_df

def plot_qini_curve(df_results, treatment_col='treatment', outcome_col='conversion'):
    """
    Plots a basic Qini curve to evaluate the uplift model's performance.
    """
    print("\n--- Generating Qini Curve ---")
    
    # Sort data by predicted uplift in descending order
    sorted_df = df_results.sort_values(by='predicted_uplift', ascending=False).reset_index(drop=True)
    
    # Cumulative counts
    sorted_df['cum_treated'] = sorted_df[treatment_col].cumsum()
    sorted_df['cum_control'] = (1 - sorted_df[treatment_col]).cumsum()
    
    # Cumulative conversions
    sorted_df['cum_treated_conv'] = (sorted_df[treatment_col] * sorted_df[outcome_col]).cumsum()
    sorted_df['cum_control_conv'] = ((1 - sorted_df[treatment_col]) * sorted_df[outcome_col]).cumsum()
    
    # Qini Calculation: Cumulative incremental conversions
    # (Treatment Conversions) - (Control Conversions * (Total Treated / Total Control))
    sorted_df['qini'] = sorted_df['cum_treated_conv'] - (sorted_df['cum_control_conv'] * (sorted_df['cum_treated'] / sorted_df['cum_control'].replace(0, 1)))
    
    # Random target line (what we'd expect without a model)
    total_treated_conv = sorted_df['cum_treated_conv'].iloc[-1]
    total_control_conv = sorted_df['cum_control_conv'].iloc[-1]
    total_treated = sorted_df['cum_treated'].iloc[-1]
    total_control = sorted_df['cum_control'].iloc[-1]
    
    overall_qini = total_treated_conv - (total_control_conv * (total_treated / total_control))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_df.index / len(sorted_df), sorted_df['qini'], label='T-Learner Uplift Model', color='blue')
    plt.plot([0, 1], [0, overall_qini], label='Random Assignment', color='red', linestyle='--')
    plt.title('Qini Curve')
    plt.xlabel('Proportion of Population Targeted')
    plt.ylabel('Incremental Conversions')
    plt.legend()
    plt.grid(True)
    plt.savefig('qini_curve.png')
    print("Saved Qini curve plot to 'qini_curve.png'")

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    # Simulating data for testing
    np.random.seed(42)
    n_samples = 20000
    
    df_mock = pd.DataFrame({
        'f1': np.random.normal(0, 1, n_samples),
        'f2': np.random.normal(0, 1, n_samples),
        'treatment': np.random.binomial(1, 0.5, n_samples)
    })
    
    # Create heterogeneous treatment effects (f1 increases uplift, f2 decreases it)
    base_prob = 0.1
    treatment_effect = 0.05 + (0.05 * df_mock['f1']) - (0.05 * df_mock['f2'])
    
    final_prob = base_prob + (df_mock['treatment'] * treatment_effect)
    df_mock['conversion'] = np.random.binomial(1, np.clip(final_prob, 0, 1))
    
    features = ['f1', 'f2']
    
    # 1. Train Model
    results_df, t_mod, c_mod = train_t_learner(df_mock, features)
    
    # 2. Segment Users
    segment_df = segment_users(results_df)
    
    # 3. Evaluate with Qini Curve
    plot_qini_curve(results_df)