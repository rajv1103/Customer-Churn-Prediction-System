import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ------------------- CSS / UI -------------------
st.markdown("""
<style>
:root{
  --bg-start: #f4f9ff;
  --bg-end: #ffffff;
  --card-bg: #ffffff;
  --accent: #0b5ed7;
  --accent-soft: rgba(11,94,215,0.15);
  --text: #0a2540;
  --muted: #6b7a90;
  --glass: rgba(255,255,255,0.75);
}
@media (prefers-color-scheme: dark){
  :root{
    --bg-start:#050f1c;
    --bg-end:#091a2c;
    --card-bg:#0b1f33;
    --accent:#4dabff;
    --accent-soft: rgba(77,171,255,0.2);
    --text:#e6f2ff;
    --muted:#9fb6cf;
    --glass: rgba(15,35,55,0.75);
  }
}
.stApp{ background: radial-gradient(1200px 600px at 10% -10%, var(--accent-soft), transparent), linear-gradient(180deg, var(--bg-start), var(--bg-end)); color: var(--text); font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont; }
.block-container{ max-width: 1000px; padding: 40px 32px; margin-left: auto; margin-right: auto; }
.header{ display:flex; align-items:auto; justify-content:center; gap:24px; margin:32px; }
.logo{ width:72px; height:72px; border-radius:100%; font-size:22px; display:flex; align-items:center; justify-content:center; font-weight:800; color:white; background: rgb(255,255,255); box-shadow: 0 18px 40px rgba(0,0,0,0.25); }
.title{ font-size:50px; font-weight:1000; letter-spacing:-0.5px; color: var(--accent); }
.subtitle{ font-size:20px; color: var(--muted); margin-top:6px; }
.stTextInput input, .stNumberInput input, .stSelectbox div[role="combobox"] input{ height:52px !important; font-size:20px !important; border-radius:12px !important; padding:10px 14px; background: rgba(11,94,215,0.05) !important; }
.stSlider > div > div > div{ height:45px; }
.stButton > button{ height:52px; border-radius:14px; font-size:16px; font-weight:700; background: linear-gradient(90deg, var(--accent), #113c8f) !important; box-shadow: 0 14px 30px rgba(110,94,215,0.45); }
.stMetric .value{ font-size:50px !important; font-weight:800 !important; }
.stMetric .delta{ font-size:20px !important; }
@media (max-width: 900px){ .block-container{padding:28px;} .title{font-size:30px;} .logo{width:60px;height:60px;} }
form { display: flex; flex-direction: column; align-items: center; }
.stTextInput, .stNumberInput, .stSelectbox, .stSlider { width: 420px !important; max-width: 100%; }
.stTextInput input, .stNumberInput input, .stSelectbox div[role="combobox"] input { width: 420px !important; margin-left: auto; margin-right: auto; }
.footer { text-align: center; margin-top: 40px; font-size: 40px; color: var(--muted); }
</style>
""", unsafe_allow_html=True)

# ------------------- Load Models -------------------
@st.cache_resource
def load_joblib_model(path="model.pkl"):
    return joblib.load(path)

@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return joblib.load(f)

# Load prediction model & encoders/scaler
model = load_joblib_model("model.pkl")
label_encoder_gender = load_pickle("label_encoder_gender.pkl")
onehot_encoder_geo = load_pickle("onehot_encoder_geo.pkl")
scaler = load_pickle("scaler.pkl")

# ------------------- Helper Functions -------------------
def safe_label_transform(encoder, value):
    try:
        return int(encoder.transform([value])[0])
    except Exception:
        return int(0)

def build_input_df(values_dict, geo_encoder):
    base = pd.DataFrame(values_dict, index=[0])
    geo_encoded = geo_encoder.transform([[values_dict["Geography"]]]).toarray()
    geo_cols = geo_encoder.get_feature_names_out(["Geography"])
    geo_df = pd.DataFrame(geo_encoded, columns=geo_cols)
    df = pd.concat([base.reset_index(drop=True), geo_df], axis=1)
    if "Geography" in df.columns:
        df = df.drop(columns=["Geography"])
    return df

def pretty_percentage(p):
    return f"{p*100:.1f}%"

# ------------------- App Header -------------------
st.markdown('<div class="header"><div class="logo">🛡️</div><div><div class="title">Customer Churn Predictor</div></div></div>', unsafe_allow_html=True)

# ------------------- Input Form -------------------
with st.form("input_form"):
    st.subheader("Customer profile")
    geography = st.selectbox("Geography", options=list(onehot_encoder_geo.categories_[0]), index=0)
    gender = st.selectbox("Gender", options=list(label_encoder_gender.classes_), index=0)
    age = st.slider("Age", min_value=18, max_value=92, value=25)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)
    balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=100.0, format="%.2f")
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=100.0, format="%.2f")
    tenure = st.slider("Tenure (years)", min_value=0, max_value=10, value=2)
    num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=1)
    has_cr_card = st.selectbox("Has Credit Card", options=[1,0], format_func=lambda x: "Yes" if x==1 else "No")
    is_active_member = st.selectbox("Is Active Member", options=[1,0], format_func=lambda x: "Yes" if x==1 else "No")
    st.markdown('</div>', unsafe_allow_html=True)
    submitted = st.form_submit_button("Predict")

# ------------------- Prediction & Display -------------------
if submitted:
    if credit_score < 300 or credit_score > 850:
        st.error("Credit Score must be between 300 and 850.")
    elif balance < 0 or estimated_salary < 0:
        st.error("Balance and Estimated Salary must be non-negative.")
    else:
        gender_encoded = safe_label_transform(label_encoder_gender, gender)
        values = {
            "CreditScore": credit_score,
            "Gender": gender_encoded,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_of_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active_member,
            "EstimatedSalary": estimated_salary,
            "Geography": geography,
        }

        input_df = build_input_df(values, onehot_encoder_geo)

        try:
            input_scaled = scaler.transform(input_df)
        except Exception as e:
            st.exception(f"Error scaling input. Check columns. Details: {e}")
            st.stop()

        # ------------------- MODEL PREDICTION -------------------
        pred = model.predict(input_scaled)
        if pred.shape[-1] == 1:
            proba = float(pred[0][0])
        else:
            proba = float(pred[0][1])
        churn = proba > 0.5

        # ------------------- Prediction Cards -------------------
        cols = st.columns([2,2,2])
        with cols[0]:
            st.subheader("Prediction")
            st.metric(label="Churn Risk", value=("High" if churn else "Low"), delta=pretty_percentage(proba))
        with cols[1]:
            st.subheader("Probability")
            st.progress(proba)
            st.write(f"**Churn probability:** {proba:.2%}")
        with cols[2]:
            st.subheader("Recommended action")
            if churn:
                st.warning("Customer at risk. Consider retention offers: personalized discount, loyalty benefits, or outreach.")
            else:
                st.success("Customer appears stable. Continue engagement and cross-sell opportunities.")

        # ------------------- Input Summary -------------------
        st.subheader("Input Summary")
        scols = st.columns(3)
        scols[0].write(f"**Geography:** {geography}")
        scols[0].write(f"**Gender:** {gender}")
        scols[0].write(f"**Age:** {age}")
        scols[1].write(f"**Credit Score:** {credit_score}")
        scols[1].write(f"**Balance:** ₹{balance:,.2f}")
        scols[1].write(f"**Estimated Salary:** ₹{estimated_salary:,.2f}")
        scols[2].write(f"**Tenure:** {tenure} years")
        scols[2].write(f"**Products:** {num_of_products}")
        scols[2].write(f"**Has Credit Card:** {'Yes' if has_cr_card==1 else 'No'}")
        scols[2].write(f"**Active Member:** {'Yes' if is_active_member==1 else 'No'}")

        # ------------------- Feature Impact -------------------
        impacts = {}
        impacts["Low Credit Score"] = max(0, (700 - credit_score) / 400)
        impacts["High Balance"] = max(0, min(1, balance / (balance + estimated_salary + 1)))
        impacts["Low Activity"] = 1.0 if is_active_member == 0 else 0.0
        impacts["Few Products"] = max(0, (2 - num_of_products) / 2)
        impacts["Short Tenure"] = max(0, (2 - tenure) / 2)

        df = pd.DataFrame({
            "feature": list(impacts.keys()),
            "impact": list(impacts.values())
        }).sort_values("impact", ascending=False).reset_index(drop=True)

        overall_risk = df["impact"].mean()

        left, right = st.columns([1, 2])
        with left:
            st.subheader("Summary")
            st.metric("Overall Risk Score", f"{overall_risk:.2f}")
            st.markdown(f"<div class='muted'>Top driver: <b>{df.loc[0,'feature']}</b> ({df.loc[0,'impact']:.2f})</div>", unsafe_allow_html=True)

        with right:
            st.subheader("Feature Impact")
            fig = px.bar(df, x="impact", y="feature", orientation="h",
                         color="impact",
                         color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"],
                         range_x=[0, 1],
                         text=df["impact"].apply(lambda v: f"{v:.2f}"))
            fig.update_traces(textposition="outside",
                              hovertemplate="<b>%{y}</b><br>Impact: %{x:.2f}<extra></extra>")
            fig.update_layout(height=420, margin=dict(l=140, r=20, t=30, b=30), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

# ------------------- Footer -------------------
st.markdown('<div class="footer">Made with ❤️ by Raaj</div>', unsafe_allow_html=True)
