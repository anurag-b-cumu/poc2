import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class ProductReturnPredictor:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df['Transaction Type'] = self.df['Transaction Type'].apply(lambda x: x == "PURCHASE")
        self.df['Returned'] = self.df['Return Reason'].notnull().astype(int)
        self.df.drop(columns=['Customer ID', 'Customer Name', 'Order ID', 'Product Name', 'Return Reason', 'Transaction Date', 'Product Category'], inplace=True)
        self.df = pd.get_dummies(self.df, columns=['Product ID'])

    def list_prod(self):
        return [col.replace("Product ID_", "") for col in self.df.columns if col.startswith("Product ID_")]
    
    def train_model(self):
        self.y = self.df['Returned']
        self.X = self.df.drop(columns=['Returned'])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        rf_accuracy = accuracy_score(y_test, y_pred)
        rf_report = classification_report(y_test, y_pred)

        return f"{rf_accuracy:.2f}"

    def get_return_probability(self, product_id):
        product_features = np.zeros((1, self.X.shape[1]))  # Initialize empty feature vector
        for i, col in enumerate(self.X.columns):
            if col == f"Product ID_{product_id}":
                product_features[0, i] = 1  # Activate selected product
                break
        
        product_features_scaled = scaler.transform(product_features)
        return_prob = self.model.predict_proba(product_features_scaled)[0][1]
        return f"{return_prob*100:.2f}%"