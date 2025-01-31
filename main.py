import streamlit as st
from backend import ProductReturnPredictor

# Streamlit app title
st.title("📦 Product Return Prediction")

# File upload
uploaded_file = "retail_sales_data.csv"

predictor = ProductReturnPredictor(uploaded_file)

# Get product list for selection
product_list = predictor.list_prod()

# Display training progress
st.subheader("🎯 Training Random Forest Model...")
rf_accuracy = predictor.train_model()
st.success(f"✅ Model trained with {rf_accuracy}% accuracy!")

# Select a product to predict return probability
st.subheader("🔎 Select a product to predict return probability")
selected_product_rf = st.selectbox("🛍️ Choose a product:", product_list, key="rf_product")

# Display prediction result
if selected_product_rf:
    return_prob = predictor.get_return_probability(selected_product_rf)
    st.header("📊 Prediction Result")
    st.subheader(f'🔄 Return Probability: {return_prob}')
