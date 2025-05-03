import streamlit as st
import pandas as pd
import numpy as np
from utils import load_data, preprocess_data, plot_fraud_distribution, plot_correlation_heatmap
from fraudmodel import train_models, predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.sidebar.title("🔍 Navigation")
app_mode = st.sidebar.radio("Go to", ["🏠 Home", "📊 Static Fraud Detection", "⚡ Dynamic Fraud Detection", "ℹ️ About"])

# ------------------------- 🏠 Home -------------------------
if app_mode == "🏠 Home":
    st.title("💳 Financial Fraud Detection System")
    st.image("images/fraud.jpg", use_column_width=True)
    st.markdown("""
    This app helps detect fraudulent financial transactions using machine learning techniques.

    **Features:**
    - Upload your own datasets
    - Visualize data distribution and patterns
    - Train and test multiple ML models
    - Compare original vs new transactions
    - Manual fraud prediction with user input
    """)

# ------------------------- 📊 Static Fraud Detection -------------------------
elif app_mode == "📊 Static Fraud Detection":
    st.title("📊 Static Fraud Detection")

    st.subheader("📂 Upload Original Dataset")
    orig_file = st.file_uploader("Upload the dataset used to train your model", type=["csv"], key="orig")

    st.subheader("📂 Upload New Dataset to Compare")
    compare_file = st.file_uploader("Upload a new transaction dataset to compare", type=["csv"], key="compare")

    if orig_file:
        st.markdown("### ✅ Original Dataset Preview")
        try:
            orig_data = load_data(orig_file)
            if orig_data is None or orig_data.empty:
                st.error("⚠️ Failed to load dataset. Please ensure it's a valid CSV file.")
            else:
                if 'id' in orig_data.columns:
                    orig_data = orig_data.drop(columns=['id'])
                st.dataframe(orig_data.head(), use_container_width=True)

                st.markdown("### 📊 Fraud vs Non-Fraud Distribution (Original)")
                plot_fraud_distribution(orig_data)

                st.markdown("### 🔍 Correlation Heatmap")
                plot_correlation_heatmap(orig_data)

                st.markdown("### 📈 Histogram: Transaction Amount")
                fig1, ax1 = plt.subplots()
                sns.histplot(orig_data['Amount'], bins=50, kde=True, ax=ax1)
                st.pyplot(fig1)

                st.markdown("### 📈 Histogram: Transaction Time")
                fig2, ax2 = plt.subplots()
                sns.histplot(orig_data['Time'], bins=50, kde=True, ax=ax2)
                st.pyplot(fig2)

                # Preprocess and Train
                st.markdown("---")
                if st.button("🔄 Train Model on Original Dataset"):
                    X_resampled, y_resampled = preprocess_data(orig_data)
                    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
                    models = train_models(X_train, y_train)
                    st.session_state['models'] = models
                    st.success("✅ Models trained on original dataset.")
        except Exception as e:
            st.error(f"⚠️ Error while loading dataset: {e}")

    if compare_file:
        st.markdown("---")
        st.markdown("### 📂 New Dataset Preview")
        try:
            new_data = load_data(compare_file)
            if new_data is None or new_data.empty:
                st.error("⚠️ Failed to load new dataset. Please ensure it's a valid CSV file.")
            else:
                if 'id' in new_data.columns:
                    new_data = new_data.drop(columns=['id'])
                st.dataframe(new_data.head(), use_container_width=True)

                st.markdown("### 📊 Fraud vs Non-Fraud Distribution (New File)")
                plot_fraud_distribution(new_data)

                st.markdown("### 🔍 Visual Comparison - Amount")
                fig3, ax3 = plt.subplots()
                sns.histplot(orig_data['Amount'], bins=50, color='blue', label='Original', alpha=0.5)
                sns.histplot(new_data['Amount'], bins=50, color='orange', label='New', alpha=0.5)
                ax3.legend()
                st.pyplot(fig3)
        except Exception as e:
            st.error(f"⚠️ Error while loading new dataset: {e}")

# ------------------------- ⚡ Dynamic Fraud Detection -------------------------
elif app_mode == "⚡ Dynamic Fraud Detection":
    st.title("⚡ Dynamic Fraud Detection")
    st.markdown("Manually input transaction data to predict fraud.")

    v_features = [st.number_input(f"V{i}", format="%.5f") for i in range(1, 29)]
    amount = st.number_input("💵 Transaction Amount", format="%.2f")
    time = st.number_input("⏰ Transaction Time", format="%.2f")

    input_data = np.array([time] + v_features + [amount]).reshape(1, -1)

    data = load_data("data/creditcard_2023.csv")
    X_resampled, y_resampled = preprocess_data(data)
    scaler = StandardScaler()
    scaler.fit(X_resampled)
    input_scaled = scaler.transform(input_data)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    models = train_models(X_train, y_train)

    selected_model = st.selectbox("📌 Select Model", ["Logistic Regression", "Random Forest", "XGBoost", "Decision Tree"])

    if st.button("🎯 Predict Transaction"):
        prediction = predict(models, selected_model, input_scaled)
        if prediction[0] == 1:
            st.error("❗ ALERT: Fraudulent Transaction Detected!")
        else:
            st.success("✅ Transaction is Legitimate.")

# ------------------------- ℹ️ About -------------------------
elif app_mode == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    - **Project Title:** Financial Fraud Detection using Machine Learning and Big Data Analytics
    - **Dataset Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
    - **Developed By:** Seerat Sachdeva
    - **Tools Used:** Python, Streamlit, Scikit-learn, XGBoost, Imbalanced-learn, Matplotlib, Seaborn
    """)
