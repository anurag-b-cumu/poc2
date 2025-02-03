import streamlit as st
from backend import ProductReturnPredictor

# Streamlit app title
st.title("📦 Product Return Prediction")

# File upload
uploaded_file = "retail_sales_data.csv"

predictor = ProductReturnPredictor(uploaded_file)

st.subheader("🎯 Training Random Forest Model...")
rf_accuracy = predictor.train_model()
st.success(f"✅ Model trained with {rf_accuracy}% accuracy!")

Product, Customer = st.tabs(["Product", "Customer"])
with Product:
    product_list = predictor.list_prod()
    
    st.subheader("🔎 Select a product to predict return probability")
    selected_product_rf = st.selectbox("🛍️ Choose a product:", product_list, key="rf_product")
    
    # Display prediction result
    if selected_product_rf:
        return_prob = predictor.get_return_probability_product(selected_product_rf)
        st.header("📊 Prediction Result")
        st.subheader(f'🔄 Return Probability: {return_prob}')

with Customer:
    product_list = predictor.list_customer()
    
    st.subheader("🔎 Select a customer to predict return probability")
    selected_product_rf = st.selectbox("🛍️ Select Customer:", product_list, key="rf_product")
    
    # Display prediction result
    if selected_product_rf:
        return_prob = predictor.get_return_probability_customer(selected_product_rf)
        st.header("📊 Prediction Result")
        st.subheader(f'🔄 Return Probability: {return_prob}')
