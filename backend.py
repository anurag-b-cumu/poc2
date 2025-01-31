import streamlit as st
from backend import ProductReturnPredictor

st.title("📦 Product Return Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])
if not uploaded_file:
    st.warning("⚠️ Please upload a dataset to proceed.")
    st.stop()

# Initialize the predictor
predictor = ProductReturnPredictor(uploaded_file)

# Get product list for selection
product_list = predictor.get_product_list()

# Streamlit Tabs for Models
random_forest_tab, linear_regression_tab = st.tabs(["🌲 Random Forest", "📉 Linear Regression"])

with random_forest_tab:
    st.subheader("🎯 Training Random Forest Model...")
    rf_accuracy = predictor.train_random_forest()
    st.success(f"✅ Model trained with {rf_accuracy['Accuracy']*100:.2f}% accuracy!")

    st.subheader("🔎 Select a product to predict return probability")
    selected_product_rf = st.selectbox("🛍️ Choose a product:", product_list, key="rf_product")

    if selected_product_rf:
        rf_prediction = predictor.predict_product_return_probability(selected_product_rf)
        st.header("📊 Prediction Result")
        st.subheader(f'🔄 Return Probability: {rf_prediction*100:.2f}%')

with linear_regression_tab:
    st.subheader("🎯 Training Linear Regression Model...")
    lr_accuracy = predictor.train_linear_regression()
    st.success(f"✅ Model trained with {lr_accuracy['Accuracy']*100:.2f}% accuracy!")

    st.subheader("🔎 Select a product to predict return probability")
    selected_product_lr = st.selectbox("🛍️ Choose a product:", product_list, key="lr_product")

    if selected_product_lr:
        lr_prediction = predictor.predict_return_probability_linear(selected_product_lr)
        st.header("📊 Prediction Result")
        st.subheader(f'🔄 Return Probability: {lr_prediction*100:.2f}%')
