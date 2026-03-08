import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import time
import plotly.express as px
from transformers import pipeline
import shap
import matplotlib.pyplot as plt

# --- 1. PAGE CONFIG & SETUP ---
st.set_page_config(page_title="PredictAware Prototype", layout="wide")
st.title("🎯 PredictAware: AI Retention Hub")

# --- 2. LOAD AI MODELS (Cached for speed) ---
@st.cache_resource
def load_sentiment_model():
    # MODIFICATION 1: Real BERT Model loaded via Hugging Face pipeline
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def train_churn_model(data):
    features = ['CLV', 'Tx_Frequency', 'Engagement_Score', 'Sentiment_Score']
    X = data[features]
    y = data['Churn_Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # MODIFICATION 4: Calculate ML Evaluation Metrics
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    return model, acc, auc, features

sentiment_analyzer = load_sentiment_model()

# --- 3. GENERATE TRAINING DATA ---
@st.cache_data
def generate_and_prep_data():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'Customer_ID': [f"CUST-{i:04d}" for i in range(n)],
        'CLV': np.random.uniform(5000, 50000, n),
        'Tx_Frequency': np.random.normal(15, 5, n),
        'Engagement_Score': np.random.uniform(0, 100, n),
        'Sentiment_Score': np.random.choice([1, 3, 5], n, p=[0.3, 0.4, 0.3])
    })
    churn_prob = 1.0 - (df['Engagement_Score'] / 100)
    churn_prob = np.where(df['Sentiment_Score'] == 1, churn_prob + 0.3, churn_prob)
    df['Churn_Label'] = (churn_prob > 0.6).astype(int)
    return df

raw_data = generate_and_prep_data()
churn_model, model_acc, model_auc, feature_cols = train_churn_model(raw_data)

# --- DISPLAY ML METRICS IN SIDEBAR ---
st.sidebar.header("🧠 Engine Diagnostics")
st.sidebar.metric("LightGBM Accuracy", f"{model_acc * 100:.1f}%")
st.sidebar.metric("Model ROC-AUC", f"{model_auc:.2f}")
st.sidebar.info("The BERT sentiment model is active and running locally.")

# --- 4. PREDICT ON DATASET ---
def predict_portfolio(df, model):
    probs = model.predict_proba(df[feature_cols])[:, 1]
    df['Churn_Risk_%'] = np.round(probs * 100, 1)
    conditions = [df['Churn_Risk_%'] > 70, df['Churn_Risk_%'] > 40]
    choices = ['Urgent RM Call + Fee Waiver', 'Automated SMS: 10% Cashback']
    df['Next_Best_Action'] = np.select(conditions, choices, default='Monitor')
    return df.sort_values('Churn_Risk_%', ascending=False)

scored_data = predict_portfolio(raw_data.copy(), churn_model)

# --- Function to highlight high-risk rows ---
def highlight_risk(row):
    color = 'background-color: #FFCCCC' if row['Churn_Risk_%'] > 70 else ''
    return [color]*len(row)

# --- 5. DASHBOARD UI ---
# MODIFICATION 3: Split into 3 streamlined tabs
tab1, tab2, tab3 = st.tabs(["📊 Portfolio Overview", "⚡ Individual Risk Simulator", "📈 Batch & Trend Analysis"])

with tab1:
    st.subheader("High-Risk Customer Queue")
    
    risk_filter = st.slider("Show customers with risk above:", 0, 100, 50)
    filtered_df = scored_data[scored_data['Churn_Risk_%'] >= risk_filter]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers at Risk", len(filtered_df))
    col2.metric("Average Churn Probability", f"{filtered_df['Churn_Risk_%'].mean():.1f}%")
    col3.metric("CLV at Risk", f"₹ {filtered_df['CLV'].sum():,.0f}")
    
    st.dataframe(filtered_df.style.apply(highlight_risk, axis=1), use_container_width=True)
    
    st.divider()
    
    # MODIFICATION 5: Plotly charts side-by-side
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig = px.histogram(scored_data, x='Churn_Risk_%', nbins=10,
                           title="Customer Churn Risk Distribution",
                           color_discrete_sequence=['#EF553B'])
        st.plotly_chart(fig, use_container_width=True)
        
    with chart_col2:
        nba_counts = scored_data['Next_Best_Action'].value_counts()
        fig2 = px.pie(values=nba_counts.values, names=nba_counts.index, title="Next Best Actions Distribution")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Interactive Risk Simulator")
    st.markdown("Test how new customer interactions instantly update the LightGBM churn prediction.")
    
    sim_col1, sim_col2 = st.columns(2)
    
    with sim_col1:
        st.markdown("**1. Current Customer Profile**")
        clv_input = st.number_input("Customer Lifetime Value (₹)", value=25000)
        tx_input = st.slider("Monthly Transactions", 0, 50, 10)
        eng_input = st.slider("Digital Engagement Score", 0, 100, 40)
        
        st.markdown("**2. Log New Support Ticket**")
        ticket_text = st.text_area("Customer Message:", value="The app is completely broken and I am tired of these high fees!")
    
    with sim_col2:
        if st.button("Analyze & Update Risk", type="primary"):
            with st.spinner("Analyzing sentiment and predicting risk..."):
                time.sleep(0.5)
                
                # Real NLP Processing
                bert_result = sentiment_analyzer(ticket_text)[0]
                sentiment_label = bert_result['label']
                sentiment_confidence = bert_result['score']
                numeric_sentiment = 1 if sentiment_label == 'NEGATIVE' else 5
                
                input_features = pd.DataFrame([[clv_input, tx_input, eng_input, numeric_sentiment]], 
                                              columns=feature_cols)
                new_risk = churn_model.predict_proba(input_features)[0][1] * 100
                
                st.markdown("### AI Assessment Results")
                st.info(f"**Sentiment Detected:** {sentiment_label} (Confidence: {sentiment_confidence*100:.1f}%)")
                
                if new_risk > 70:
                    st.error(f"**Updated Churn Risk:** {new_risk:.1f}% 🚨")
                    st.warning("**Next Best Action:** Route to Escalation Team Immediately.")
                else:
                    st.success(f"**Updated Churn Risk:** {new_risk:.1f}% ✅")
                    st.success("**Next Best Action:** Continue standard monitoring.")
                
                # MODIFICATION 2: SHAP Explainability Plot
                
                st.markdown("### Explainable AI (Why this score?)")
                explainer = shap.TreeExplainer(churn_model)
                shap_values = explainer(input_features)
                
                fig, ax = plt.subplots(figsize=(6, 4))
                shap.plots.waterfall(shap_values[0], show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.clf()

with tab3:
    st.subheader("Batch Message Analysis")
    batch_messages = st.text_area("Enter multiple messages (one per line):", value="Bad service\nLoving the app\nFees too high")
    
    if st.button("Analyze Batch Messages"):
        results = []
        for msg in batch_messages.splitlines():
            sent = sentiment_analyzer(msg)[0]
            score = 1 if sent['label'] == 'NEGATIVE' else 5
            df_input = pd.DataFrame([[25000, 10, 40, score]], columns=feature_cols)
            risk = churn_model.predict_proba(df_input)[0][1]*100
            results.append({'Message': msg, 'BERT Sentiment': sent['label'], 'Churn_Risk_%': round(risk,1)})
        st.table(pd.DataFrame(results))
        
    st.divider()
    st.subheader("Simulated Engagement Impact")
    eng_slider = st.slider("Set Engagement Score for All Customers:", 0, 100, 40)
    temp_df = raw_data.copy()
    temp_df['Engagement_Score'] = eng_slider
    temp_df = predict_portfolio(temp_df, churn_model)
    st.dataframe(temp_df[['Customer_ID','Churn_Risk_%','Engagement_Score']], use_container_width=True)