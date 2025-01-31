import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class ProductReturnPredictor:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.process_data()
        self.model_rf = None
        self.model_lr = None

    def process_data(self):
        """Preprocess dataset: feature engineering & encoding"""
        self.df['Transaction Type'] = self.df['Transaction Type'].apply(lambda x: x == "PURCHASE")
        self.df['Product'] = f"{self.df['Product ID']} - {self.df['Product Name']}"
        self.df.rename(columns={'Product Category': 'Category'}, inplace=True)
        self.df['Returned'] = self.df['Return Reason'].notnull().astype(int)

        # Drop unnecessary columns
        self.df.drop(columns=['Customer ID', 'Customer Name', 'Order ID', 'Product ID', 'Product Name', 'Return Reason'], inplace=True)

        # Date feature extraction
        self.df['Transaction Date'] = pd.to_datetime(self.df['Transaction Date'])
        self.df['Transaction Year'] = self.df['Transaction Date'].dt.year
        self.df['Transaction Month'] = self.df['Transaction Date'].dt.month
        self.df['Transaction Day'] = self.df['Transaction Date'].dt.day
        self.df.drop(columns=['Transaction Date'], inplace=True)

        # Encode categorical variables
        le = LabelEncoder()
        self.df['Category'] = le.fit_transform(self.df['Category'])
        self.df['Product'] = le.fit_transform(self.df['Product'])

        # Define features and target
        self.X = self.df.drop(columns=['Returned'])
        self.y = self.df['Returned']

        # Scale numerical features
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

    def train_random_forest(self):
        """Train a Random Forest Classifier"""
        X_train, X_test, y_train, y_test = train_test_split(self.X_scaled, self.y, test_size=0.2, random_state=42, stratify=self.y)
        self.model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_rf.fit(X_train, y_train)

        accuracy = self.model_rf.score(X_test, y_test)
        return {"Accuracy": accuracy}

    def train_linear_regression(self):
        """Train a Logistic Regression model"""
        X_train, X_test, y_train, y_test = train_test_split(self.X_scaled, self.y, test_size=0.2, random_state=42, stratify=self.y)
        self.model_lr = LogisticRegression()
        self.model_lr.fit(X_train, y_train)

        accuracy = self.model_lr.score(X_test, y_test)
        return {"Accuracy": accuracy}

    def get_product_list(self):
        """Return list of unique products"""
        return self.df['Product'].unique().tolist()

    def predict_product_return_probability(self, product):
        """Predict return probability using Random Forest"""
        product_index = self.df[self.df['Product'] == product].index[0]
        features = np.array(self.X_scaled[product_index]).reshape(1, -1)
        return self.model_rf.predict_proba(features)[0][1] if self.model_rf else None

    def predict_return_probability_linear(self, product):
        """Predict return probability using Logistic Regression"""
        product_index = self.df[self.df['Product'] == product].index[0]
        features = np.array(self.X_scaled[product_index]).reshape(1, -1)
        return self.model_lr.predict_proba(features)[0][1] if self.model_lr else None