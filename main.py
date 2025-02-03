import streamlit as st
from backend import ProductReturnPredictor

# Streamlit app title
st.title("📦 Product Return Prediction")

# File upload
uploaded_file = "retail_sales.csv"  # Placeholder for simplicity

if uploaded_file:
    predictor = ProductReturnPredictor(uploaded_file)

    st.subheader("🎯 Training Random Forest Model...")
    rf_accuracy = predictor.train_model()
    st.success(f"✅ Model trained with {rf_accuracy}% accuracy!")

    # Tabs for Product and Customer
    Product, Customer = st.tabs(["Product", "Customer"])

    with Product:
        product_list = predictor.list_prod()

        st.subheader("🔎 Select a product to predict return probability")
        selected_product = st.selectbox("🛍️ Choose a product:", product_list, key="product")

        if selected_product:
            return_prob = predictor.get_return_probability_product(selected_product)
            st.header("📊 Prediction Result")
            st.subheader(f'🔄 Return Probability: {return_prob}')

    with Customer:
        customer_list = predictor.list_customer()

        st.subheader("🔎 Select a customer to predict return probability")
        selected_customer = st.selectbox("🛍️ Select Customer:", customer_list, key="customer")

        if selected_customer:
            return_prob = predictor.get_return_probability_customer(selected_customer)
            st.header("📊 Prediction Result")
            st.subheader(f'🔄 Return Probability: {return_prob}')
else:
    st.warning("Please upload a valid dataset to continue.")
