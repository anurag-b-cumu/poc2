import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class ProductReturnPredictor:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.preprocess_data()

    def preprocess_data(self):
        # Basic data cleaning
        self.df['Transaction Type'] = self.df['Transaction Type'].apply(lambda x: x == "PURCHASE")
        self.df['Returned'] = self.df['Return Reason'].notnull().astype(int)

        # Dropping irrelevant columns
        self.df.drop(columns=['Customer Name', 'Order ID', 'Product Name', 'Return Reason', 
                               'Transaction Date', 'Product Category'], inplace=True)

        # Encoding categorical variables
        self.df = pd.get_dummies(self.df, columns=['Product ID', 'Customer ID'])

    def list_prod(self):
        return [col.replace("Product ID_", "") for col in self.df.columns if col.startswith("Product ID_")]

    def list_customer(self):
        return [col.replace("Customer ID_", "") for col in self.df.columns if col.startswith("Customer ID_")]

    def train_model(self):
        self.y = self.df['Returned']
        self.X = self.df.drop(columns=['Returned'])

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, y_pred)

        return f"{rf_accuracy * 100:.2f}"

    def get_return_probability(self, identifier, id_type='Product'):
        feature_name = f"{id_type} ID_{identifier}"
        if feature_name not in self.X.columns:
            return "N/A"

        # Create feature vector
        feature_vector = np.zeros((1, self.X.shape[1]))
        idx = self.X.columns.get_loc(feature_name)
        feature_vector[0, idx] = 1

        feature_vector_scaled = self.scaler.transform(feature_vector)
        return_prob = self.model.predict_proba(feature_vector_scaled)[0][1]
        return f"{return_prob * 100:.2f}%"

    def get_return_probability_product(self, product_id):
        return self.get_return_probability(product_id, id_type='Product')

    def get_return_probability_customer(self, customer_id):
        return self.get_return_probability(customer_id, id_type='Customer')
