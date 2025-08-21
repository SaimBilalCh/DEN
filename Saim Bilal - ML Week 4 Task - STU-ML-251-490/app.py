import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from ml_utils import train_and_evaluate

st.set_page_config(page_title="Breast Cancer ML Comparison", layout="wide")

st.title("🔬 Breast Cancer ML Model Comparison")
st.markdown("Compare **Logistic Regression** and **Random Forest** with/without Feature Selection.")

#Sidebar
st.sidebar.header("⚙️ Settings")
k = st.sidebar.slider("Number of Features (SelectKBest)", 5, 20, 10)

#Run ML pipeline
with st.spinner("Training models..."):
    results, selected_features = train_and_evaluate(k)

st.sidebar.write("✅ Selected Features:", selected_features)

#Convert results to DataFrame
df_results = pd.DataFrame(results).T
st.subheader("📊 Model Performance Metrics")
st.dataframe(df_results)

#Visualization
st.subheader("📈 Metrics Comparison")

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
fig, ax = plt.subplots(figsize=(8, 5))
df_results[metrics].plot(kind="bar", ax=ax)
plt.xticks(rotation=45, ha="right")
plt.title("Model Metrics Comparison")
plt.ylabel("Score")
st.pyplot(fig)

#Confusion Matrix
st.subheader("📉 Confusion Matrices")
cols = st.columns(2)
for i, (name, vals) in enumerate(results.items()):
    cm = vals["Confusion Matrix"]
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    cols[i % 2].pyplot(fig)

st.success("✅ Models trained, results saved to `ml_results.json`, models saved to `models.pkl`.")
