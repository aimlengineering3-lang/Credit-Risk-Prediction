import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="PakCredit AI",
    page_icon="🏦",
    layout="wide"
)

# ==========================================================
# LOAD FILES
# ==========================================================

pipeline = joblib.load("models/credit_risk_pipeline.pkl")
metrics = joblib.load("models/metrics.pkl")
feature_importance = joblib.load("models/feature_importance.pkl")

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background:
    radial-gradient(circle at top left,#1e3a8a,#0f172a 35%),
    #020617;
}

[data-testid="stSidebar"]{
    background:#0f172a;
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

.hero{
    background:linear-gradient(
    135deg,
    #2563eb,
    #1e293b
    );

    padding:40px;
    border-radius:25px;
    color:white;
    margin-bottom:25px;
    box-shadow:0px 10px 30px rgba(0,0,0,0.4);
}

.hero h1{
    color:white;
    font-size:58px;
    margin-bottom:5px;
}

.hero h3{
    color:#e2e8f0;
}

.hero p{
    color:#cbd5e1;
}

.kpi-card{
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.1);
    backdrop-filter:blur(10px);
    border-radius:20px;
    padding:20px;
    text-align:center;
    margin-bottom:15px;
}

.kpi-title{
    color:#cbd5e1;
    font-size:15px;
}

.kpi-value{
    font-size:34px;
    font-weight:700;
    color:#60a5fa;
}

.section-card{
    background:#0f172a;
    border:1px solid #334155;
    border-radius:20px;
    padding:20px;
    margin-top:20px;
    margin-bottom:20px;
}

.section-card h2{
    color:white;
}

.result-card{
    background:linear-gradient(
    135deg,
    #0f172a,
    #1e293b
    );

    border-radius:25px;
    padding:35px;
    text-align:center;
    border:1px solid #475569;
}

.score{
    font-size:90px;
    font-weight:900;
    color:#38bdf8;
}

.low{
    color:#22c55e;
}

.medium{
    color:#f59e0b;
}

.high{
    color:#ef4444;
}

.recommendation-card{
    background:#111827;
    border-radius:15px;
    padding:20px;
    border:1px solid #374151;
    margin-top:15px;
}

.main-factor{
    background:#111827;
    border-radius:12px;
    padding:10px;
    margin-bottom:8px;
    border:1px solid #374151;
}
            
/* =====================================================
   GLOBAL TEXT FIX (ALL WIDGETS)
===================================================== */

body, .stApp {
    color: #e5e7eb !important;
}

/* ALL LABELS (THIS IS THE KEY FIX) */
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* Streamlit default label container */
[data-testid="stWidgetLabel"] {
    color: #e2e8f0 !important;
}

/* =====================================================
   INPUT BOX FIX (TEXT + DROPDOWNS)
===================================================== */

input, textarea {
    color: #0f172a !important;
    font-weight: 500;
}

/* BaseWeb selectbox (CRITICAL FIX) */
div[data-baseweb="select"] > div {
    color: #0f172a !important;
}

/* Selected value text */
div[data-baseweb="select"] span {
    color: #0f172a !important;
}

/* Dropdown menu items */
div[data-baseweb="menu"] li {
    color: #0f172a !important;
}

/* Prevent truncation of long labels */
div[data-baseweb="select"] {
    white-space: normal !important;
    overflow: visible !important;
}

/* =====================================================
   SLIDER FIX
===================================================== */

.stSlider > div {
    color: #e2e8f0 !important;
}

/* slider value text */
.stSlider [data-testid="stTickBarMin"], 
.stSlider [data-testid="stTickBarMax"] {
    color: #94a3b8 !important;
}

/* =====================================================
   SECTION HEADERS FIX
===================================================== */

h1, h2, h3, h4 {
    color: white !important;
}

/* =====================================================
   COLUMN ALIGNMENT FIX (IMPORTANT)
===================================================== */

[data-testid="column"] {
    padding-top: 5px;
}

/* Prevent overlap / squeezing */
.element-container {
    margin-bottom: 8px;
}
            


</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🏦 PakCredit AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Credit Assessment",
        "Portfolio Analytics"
    ]
)

# ==========================================================
# USER FRIENDLY LABELS
# ==========================================================

account_map = {
    "Negative Balance": "< 0 DM",
    "Low Balance": "0 to < 200 DM",
    "Healthy Balance": ">= 200 DM",
    "No Checking Account": "no checking account"
}

savings_map = {
    "No Savings": "unknown/ no savings account",
    "Very Low Savings": "< 100 DM",
    "Moderate Savings": "100 to < 500 DM",
    "Good Savings": "500 to < 1000 DM",
    "Strong Savings": ">= 1000 DM"
}

employment_map = {
    "Unemployed": "unemployed",
    "Less Than 1 Year": "< 1 year",
    "1 - 4 Years": "1 to < 4 years",
    "4 - 7 Years": "4 to < 7 years",
    "7+ Years": ">= 7 years"
}

# ==========================================================
# CREDIT ASSESSMENT PAGE
# ==========================================================

if page == "Credit Assessment":

    st.markdown("""
    <div class='hero'>
        <h1>🏦 PakCredit AI</h1>
        <h3>Smart Credit Assessment Platform</h3>
        <p>Helping credit officers make informed financing decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI CARDS

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Model Accuracy</div>
            <div class='kpi-value'>
                {metrics['accuracy']:.0%}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='kpi-card'>
            <div class='kpi-title'>Risk Engine</div>
            <div class='kpi-value'>RF-500</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Training Records</div>
            <div class='kpi-value'>
                {metrics['training_records']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class='kpi-card'>
            <div class='kpi-title'>System Status</div>
            <div class='kpi-value'>Active</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='section-card'>
        <h2>👤 Applicant Information</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider(
            "Applicant Age",
            18,
            75,
            35
        )

    with col2:
        residence_since = st.slider(
            "Years At Current Residence",
            1,
            10,
            3
        )

    with col3:
        housing = st.selectbox(
            "Housing Status",
            [
                "own",
                "rent",
                "for free"
            ]
        )

    st.markdown("""
    <div class='section-card'>
        <h2>💳 Financial Information</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:

        account_label = st.selectbox(
            "Bank Account Status",
            list(account_map.keys())
        )

        savings_label = st.selectbox(
            "Savings Profile",
            list(savings_map.keys())
        )

        employment_label = st.selectbox(
            "Employment Stability",
            list(employment_map.keys())
        )

    with col2:

        n_credits = st.slider(
            "Existing Financing Accounts",
            1,
            5,
            1
        )

        payment_to_income_ratio = st.slider(
            "Installment To Income Ratio",
            1,
            4,
            2
        )

        telephone = st.selectbox(
            "Telephone Availability",
            [
                "yes, registered under the customers name",
                "none"
            ]
        )

    with col3:

        collateral = st.selectbox(
            "Collateral Type",
            [
                "real estate",
                "car",
                "savings agreement/life insurance",
                "none"
            ]
        )

        job = st.selectbox(
            "Occupation Category",
            [
                "management/ self-employed/highly qualified employee",
                "skilled employee/ official",
                "unskilled - resident",
                "unemployed/ unskilled - non-resident"
            ]
        )

        other_installment_plans = st.selectbox(
            "Other Installment Plans",
            [
                "none",
                "bank",
                "stores"
            ]
        )

    st.markdown("""
    <div class='section-card'>
        <h2>🏦 Financing Details</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_amount = st.number_input(
            "Credit Amount Requested",
            min_value=250,
            max_value=20000,
            value=2000,
            step=250
        )

        duration = st.slider(
            "Loan Duration (Months)",
            6,
            72,
            24
        )

    with col2:
        purpose = st.selectbox(
            "Loan Purpose",
            [
                "car",
                "radio/TV",
                "furniture/equipment",
                "business",
                "education",
                "repairs",
                "vacation/others"
            ]
        )

        installment_rate = st.slider(
            "Installment Rate (% of income)",
            1,
            4,
            2
        )

    with col3:
        foreign_worker = st.selectbox(
            "Foreign Worker",
            ["yes", "no"]
        )

        existing_loans = st.slider(
            "Number of Existing Loans",
            0,
            4,
            1
        )

# ==========================================================
# PREDICTION BUTTON
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_btn = st.button("🚀 Run Credit Risk Assessment")

# ==========================================================
# PREDICTION LOGIC
# ==========================================================

def map_inputs():
    return {
        "status_account": account_map[account_label],
        "status_savings": savings_map[savings_label],
        "years_employment": employment_map[employment_label],

        "age": age,
        "residence_since": residence_since,
        "housing": housing,

        "credit_history": "good",   # or make dropdown later

        "credit_amount": credit_amount,
        "month_duration": duration,

        "purpose": purpose,
        "installment_rate": installment_rate,

        "payment_to_income_ratio": payment_to_income_ratio,
        "n_credits": n_credits,

        "other_installment_plans": other_installment_plans,
        "telephone": telephone,
        "foreign_worker": foreign_worker,

        "n_guarantors": 0,
        "secondary_obligor": "none"
    }

if predict_btn:


    # STEP 1: create input first
    input_data = pd.DataFrame([map_inputs()])

    # STEP 2: align columns
    model_columns = joblib.load("models/model_columns.pkl")
    input_data = input_data.reindex(columns=model_columns)

    # STEP 3: fill missing values
    input_data = input_data.fillna("unknown")

    # STEP 4: prediction
    prediction = pipeline.predict(input_data)[0]

    try:
        probability = pipeline.predict_proba(input_data)[0][1]
    except:
        probability = None


    # ======================================================
    # RISK CATEGORY
    # ======================================================

    if probability is not None:
        risk_score = int(probability * 100)
    else:
        risk_score = int(prediction * 100)

    if risk_score < 35:
        category = "LOW RISK"
        css_class = "low"
    elif risk_score < 70:
        category = "MEDIUM RISK"
        css_class = "medium"
    else:
        category = "HIGH RISK"
        css_class = "high"

    # ======================================================
    # RESULT DISPLAY (CINEMATIC DASHBOARD)
    # ======================================================

    st.markdown("## 📊 Assessment Result")

    st.markdown(f"""
    <div class='result-card'>

        <h2>Credit Risk Score</h2>

        <div class='score'>
            {risk_score}
        </div>

        <h3>Out of 100</h3>

        <h1 class='{css_class}'>
            {category}
        </h1>

    </div>
    """, unsafe_allow_html=True)

    # ======================================================
    # INSIGHTS PANEL
    # ======================================================

    st.markdown("### 🔍 Key Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='recommendation-card'>
        <h4>Risk Interpretation</h4>
        <p>
        The model evaluates repayment ability based on financial stability,
        credit history, and income-to-obligation ratio.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='recommendation-card'>
        <h4>Decision Support</h4>
        <p>
        Use this score as a decision support tool. Final approval should
        include manual credit officer review.
        </p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================================
# PORTFOLIO ANALYTICS PAGE
# ==========================================================

if page == "Portfolio Analytics":

    st.markdown("""
    <div class='hero'>
        <h1>📊 Portfolio Analytics</h1>
        <h3>Bank-wide Credit Risk Intelligence</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Size", f"{metrics['dataset_size']:,}")

    with col2:
        st.metric("Model Accuracy", f"{metrics['accuracy']:.0%}")

    with col3:
        st.metric("Active Model", "RF-500")

    st.markdown("### 🔑 Key Drivers Behind Risk Assessments")

    # =========================
    # FIXED FEATURE IMPORTANCE HANDLING
    # =========================

    features = feature_importance.copy()

    # If DataFrame → convert to Series safely
    if isinstance(features, pd.DataFrame):
        # assume first column = feature name, second = importance
        features.columns = features.columns.astype(str)

        # handle common structure
        if features.shape[1] >= 2:
            features = features.set_index(features.columns[0])[features.columns[1]]
        else:
            features = features.iloc[:, 0]

    features = features.sort_values(ascending=True)

    fig, ax = plt.subplots()
    ax.barh(features.index, features.values)
    ax.set_title("Feature Importance")

    st.pyplot(fig)

    st.markdown("""
    <div class='section-card'>
        <h3>System Overview</h3>
        <p>
        This dashboard provides a real-time overview of credit risk distribution
        and model behavior across the entire loan portfolio.
        </p>
    </div>
    """, unsafe_allow_html=True)