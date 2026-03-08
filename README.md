

# 🎯 PredictAware – AI Retention Hub (Prototype)

## Overview

PredictAware is an **AI-powered predictive customer outreach platform** designed to help organizations proactively identify customers at risk of churn and deploy targeted retention strategies.

This repository contains a **basic prototype built using Streamlit** to simulate how PredictAware could function in a real-world environment. The prototype demonstrates the **core AI pipeline** including churn prediction, sentiment analysis, explainable AI, and automated next-best-action recommendations.

⚠️ **Note:**
This is a **simulation prototype** created for demonstration and hackathon purposes. It uses **synthetic data and simplified workflows** to represent the core system architecture.

---

# 🚀 Key Features

### 📊 Portfolio Risk Dashboard

* Displays customers ranked by **predicted churn risk**
* Calculates **Customer Lifetime Value (CLV) at risk**
* Provides **Next Best Action recommendations**
* Includes interactive **risk distribution visualizations**

### ⚡ Interactive Risk Simulator

Allows users to simulate customer scenarios by entering:

* Customer Lifetime Value
* Transaction frequency
* Digital engagement score
* Customer support message

The system then:

1. Runs **NLP sentiment analysis**
2. Predicts **churn probability**
3. Generates **AI-based retention recommendations**
4. Explains predictions using **Explainable AI**

### 🧠 AI-Powered Sentiment Analysis

The prototype integrates a **BERT-based NLP model** to analyze customer messages and detect sentiment signals that may indicate dissatisfaction.

### 🔍 Explainable AI

Uses **SHAP (SHapley Additive Explanations)** to visualize:

* How each feature contributes to churn prediction
* Why the model assigned a specific risk score

### 📈 Batch & Trend Analysis

Simulates large-scale analysis by:

* Processing multiple customer messages
* Evaluating sentiment and churn risk
* Simulating the impact of changes in engagement levels

---

# 🧠 AI & Machine Learning Components

### Churn Prediction Model

* Algorithm: **LightGBM**
* Inputs:

  * Customer Lifetime Value (CLV)
  * Transaction Frequency
  * Engagement Score
  * Sentiment Score
* Outputs:

  * Churn probability
  * Risk classification
  * Recommended action

### NLP Sentiment Analysis

* Model: **DistilBERT**
* Used to extract **sentiment signals from customer messages**

### Explainable AI

* Framework: **SHAP**
* Provides transparency into **model predictions**

---

# 🛠️ Tech Stack

### Frontend

* Streamlit – interactive dashboard UI

### Machine Learning

* LightGBM – churn prediction
* Scikit-learn – model training utilities

### NLP

* Hugging Face Transformers – BERT sentiment analysis

### Data & Analytics

* Pandas
* NumPy

### Visualization

* Plotly
* Matplotlib
* SHAP

---

# 📂 Prototype Workflow

```
Customer Data (Simulated)
        │
        ▼
Feature Engineering
(CL V, Tx Frequency, Engagement Score, Sentiment)
        │
        ▼
Churn Prediction Model (LightGBM)
        │
        ▼
Churn Risk Score
        │
        ▼
Next Best Action Engine
        │
        ▼
Explainable AI (SHAP)
        │
        ▼
Streamlit Dashboard Visualization
```

---

# ⚙️ Running the Prototype

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/predictaware-prototype.git
cd predictaware-prototype
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

The dashboard will open in your browser.

---

# 📌 Prototype Limitations

This version is **a simplified proof-of-concept**:

* Uses **synthetic customer data**
* Models are **lightweight for demonstration**
* No real CRM integration
* No real-time data pipelines

The full system would include:

* Real-time data ingestion
* Feature store infrastructure
* enterprise CRM integration
* scalable cloud deployment

---

# 💡 Future Enhancements

Potential improvements include:

* Real customer dataset integration
* Real-time data pipelines (Kafka / Airflow)
* advanced uplift modeling
* omnichannel campaign automation
* deployment on cloud infrastructure
* full enterprise CRM integration

---

# 👥 Project Context

This prototype was developed as part of a **hackathon project exploring AI-driven customer retention systems**.

The goal was to demonstrate how **predictive analytics, NLP, and explainable AI can be combined to proactively prevent customer churn.**

---

