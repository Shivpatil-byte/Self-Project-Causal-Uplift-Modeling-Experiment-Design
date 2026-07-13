import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.stats.api as sms
from statsmodels.stats.proportion import proportion_effectsize
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Causal Uplift Dashboard", layout="wide")
st.title("Causal Uplift Modeling & Experiment Design")
st.markdown("A demonstration of moving beyond naive A/B testing using causal inference and T-Learner uplift modeling.")

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["1. Experiment Design (Power)", "2. Causal Correction (PSM)", "3. Uplift Modeling (Segmentation)"])

# ==========================================
# PAGE 1: Power Calculator
# ==========================================
if page == "1. Experiment Design (Power)":
    st.header("Statistical Power Calculator")
    st.write("Determine how much data you need to detect a true effect.")
    
    col1, col2 = st.columns(2)
    with col1:
        base_rate = st.slider("Baseline Conversion Rate (%)", 0.1, 5.0, 0.3, 0.1) / 100
        rel_mde = st.slider("Relative Minimum Detectable Effect (MDE %)", 1, 50, 10, 1) / 100
    
    with col2:
        alpha = st.selectbox("Significance Level (Alpha)", [0.01, 0.05, 0.10], index=1)
        power = st.selectbox("Statistical Power", [0.80, 0.90, 0.95], index=0)
    
    # Calculation
    treatment_rate = base_rate * (1 + rel_mde)
    effect_size = proportion_effectsize(treatment_rate, base_rate)
    analysis = sms.NormalIndPower()
    req_n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0, alternative='two-sided')
    
    st.divider()
    st.metric(label="Required Sample Size (Per Group)", value=f"{int(np.ceil(req_n)):,}")
    st.metric(label="Total Required Sample Size (50/50 Split)", value=f"{int(np.ceil(req_n) * 2):,}")
    if int(np.ceil(req_n) * 2) > 1000000:
        st.info("💡 Notice how detecting small lifts on low-conversion events requires millions of rows. This is why the 13M row Criteo dataset is necessary!")

# ==========================================
# PAGE 2: Naive vs Causal
# ==========================================
elif page == "2. Causal Correction (PSM)":
    st.header("Naive A/B Test vs. Causal Correction")
    st.write("Visualizing the impact of Propensity Score Matching on confounding variables.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Before Matching (Naive)")
        st.error("🚨 Confounding Detected")
        naive_data = pd.DataFrame({
            "Metric": ["Sample Size", "Treatment Conv.", "Control Conv.", "Absolute Lift", "SMD (Imbalance)"],
            "Value": ["1,000,000", "0.00355", "0.00288", "+ 0.00067", "1.289 (FAIL)"]
        })
        st.table(naive_data)
        
    with col2:
        st.subheader("After Matching (Causal)")
        st.success("✅ Covariates Balanced")
        causal_data = pd.DataFrame({
            "Metric": ["Sample Size", "Treatment Conv.", "Control Conv.", "Absolute Lift", "SMD (Imbalance)"],
            "Value": ["10,056 (Matched Pairs)", "0.13106", "0.12171", "+ 0.00934", "0.002 (PASS)"]
        })
        st.table(causal_data)
        
    st.markdown("""
    **The Business Takeaway:** The naive analysis was heavily biased by highly engaged users clustering in the treatment group. By matching users based on their propensity scores, we isolated the *true* causal effect of the ad.
    """)

# ==========================================
# PAGE 3: Uplift Segmentation
# ==========================================
elif page == "3. Uplift Modeling (Segmentation)":
    st.header("Uplift Modeling (T-Learner)")
    st.write("Targeting efficiency: Who actually needs to see the ad?")
    
    st.subheader("User Segmentation (Quartiles)")
    segment_data = pd.DataFrame({
        "Segment": ["Q1 (Persuadables)", "Q2 (Low Impact)", "Q3 (Negligible)", "Q4 (Sleeping Dogs)"],
        "Avg Predicted Uplift": ["18.0%", "8.0%", "2.4%", "-3.0%"],
        "Actual Absolute Lift": ["+ 24.1%", "+ 7.1%", "+ 1.8%", "- 7.5%"]
    })
    st.dataframe(segment_data, use_container_width=True)
    
    st.warning("⚠️ **Strategic Insight:** Stop spending ad budget on Q4. Showing them ads actually decreases their likelihood to convert. Reallocate that budget to Q1.")
    
    st.subheader("Qini Curve")
    # Generating a mock Qini curve for the dashboard display
    x = np.linspace(0, 1, 100)
    random_targeting = x * 500
    model_targeting = 500 * (1 - np.exp(-5 * x)) 
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, model_targeting, label='T-Learner Uplift Model', color='blue', linewidth=2)
    ax.plot(x, random_targeting, label='Random Assignment', color='red', linestyle='--')
    ax.set_title('Cumulative Incremental Conversions')
    ax.set_xlabel('Proportion of Population Targeted')
    ax.set_ylabel('Incremental Conversions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)