import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logging.basicConfig(level=logging.INFO)

def create_high_risk_target(df_transactions, reference_date=None, n_clusters=3, random_state=42):
    """RFM + KMeans to create proxy target (1 = high risk)."""
    if reference_date is None:
        reference_date = pd.to_datetime(df_transactions['TransactionStartTime']).max()
    df = df_transactions.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    monetary = df.groupby('CustomerId')['Amount'].sum().rename('Monetary')
    frequency = df.groupby('CustomerId').size().rename('Frequency')
    recency = df.groupby('CustomerId')['TransactionStartTime'].max().apply(
        lambda x: (reference_date - x).days
    ).rename('Recency')
    
    rfm = pd.DataFrame({'Monetary': monetary, 'Frequency': frequency, 'Recency': recency}).reset_index()
    scaler = MinMaxScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Monetary', 'Frequency', 'Recency']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify highest risk cluster: lowest monetary & frequency, highest recency
    cluster_means = rfm.groupby('Cluster')[['Monetary', 'Frequency', 'Recency']].mean()
    cluster_means['RiskScore'] = -cluster_means['Monetary'] - cluster_means['Frequency'] + cluster_means['Recency']
    high_risk_cluster = cluster_means['RiskScore'].idxmax()
    logging.info(f"High-risk cluster: {high_risk_cluster}")
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    return rfm[['CustomerId', 'is_high_risk']]

class FullDataProcessor:
    def init(self, reference_date=None, random_state=42):
        self.reference_date = reference_date
        self.random_state = random_state
        
    def fit_transform(self, df_raw):
        # Create target
        target_df = create_high_risk_target(df_raw, self.reference_date, random_state=self.random_state)
        # Aggregate customer features
        cust_features = self._aggregate_customer_features(df_raw)
        final = cust_features.merge(target_df, on='CustomerId', how='inner')
        final.drop('CustomerId', axis=1, inplace=True)
        return final
    
    def _aggregate_customer_features(self, df):
        df = df.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['Hour'] = df['TransactionStartTime'].dt.hour
        df['Day'] = df['TransactionStartTime'].dt.day
        df['Month'] = df['TransactionStartTime'].dt.month
        
        agg_dict = {
            'Amount': ['sum', 'mean', 'std', 'min', 'max'],
            'Hour': ['mean', 'std'],
            'Day': ['mean'],
            'Month': ['mean'],
            'ProductCategory': lambda x: x.mode()[0] if len(x.mode())>0 else 'Unknown',
            'ChannelId': lambda x: x.mode()[0] if len(x.mode())>0 else 'Unknown',
        }
        cust = df.groupby('CustomerId').agg(agg_dict)
        cust.columns = ['_'.join(col).strip() for col in cust.columns.values]
        cust.reset_index(inplace=True)
        # One-hot encode categoricals
        cust = pd.get_dummies(cust, columns=['ProductCategory_<lambda>', 'ChannelId_<lambda>'], 
                              prefix=['Cat', 'Channel'])
        # Fill NaN from std if only one transaction
        cust.fillna(0, inplace=True)
        return cust

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

class FeatureEngineer:
    """Handles all feature engineering from raw transaction data."""
    
    def init(self):
        pass
    
    def create_customer_features(self, df):
        """
        Input: raw transaction DataFrame
        Output: DataFrame with one row per customer and aggregated features
        """
        df = df.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Time features
        df['Hour'] = df['TransactionStartTime'].dt.hour
        df['Day'] = df['TransactionStartTime'].dt.day
        df['Month'] = df['TransactionStartTime'].dt.month
        
        # Aggregation dictionary
        agg_dict = {
            'Amount': ['sum', 'mean', 'std', 'min', 'max'],
            'Hour': ['mean', 'std'],
            'Day': ['mean'],
            'Month': ['mean'],
            'ProductCategory': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
            'ChannelId': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
        }
        
        cust = df.groupby('CustomerId').agg(agg_dict)
        # Flatten multi-level column names
        cust.columns = ['_'.join(col).strip() for col in cust.columns.values]
        cust.reset_index(inplace=True)
        
        # One-hot encode the two categorical columns (mode values)
        cust = pd.get_dummies(cust, columns=['ProductCategory_<lambda>', 'ChannelId_<lambda>'],
                              prefix=['Cat', 'Channel'])
        # Fill any NaN (e.g., std for customers with only 1 transaction)
        cust.fillna(0, inplace=True)
        
        return cust
    

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def create_rfm_target(df_transactions, reference_date=None, n_clusters=3, random_state=42):
    """
    Compute RFM, cluster customers, and return a DataFrame with CustomerId and is_high_risk.
    """
    if reference_date is None:
        reference_date = pd.to_datetime(df_transactions['TransactionStartTime']).max()
    
    df = df_transactions.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # RFM calculations
    monetary = df.groupby('CustomerId')['Amount'].sum().rename('Monetary')
    frequency = df.groupby('CustomerId').size().rename('Frequency')
    recency = df.groupby('CustomerId')['TransactionStartTime'].max().apply(
        lambda x: (reference_date - x).days
    ).rename('Recency')
    
    rfm = pd.DataFrame({'Monetary': monetary, 'Frequency': frequency, 'Recency': recency}).reset_index()
    
    # Scale RFM
    scaler = MinMaxScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Monetary', 'Frequency', 'Recency']])
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify high-risk cluster: lowest Monetary & Frequency, highest Recency
    cluster_summary = rfm.groupby('Cluster')[['Monetary', 'Frequency', 'Recency']].mean()
    cluster_summary['RiskScore'] = -cluster_summary['Monetary'] - cluster_summary['Frequency'] + cluster_summary['Recency']
    high_risk_cluster = cluster_summary['RiskScore'].idxmax()
    logging.info(f"High-risk cluster identified: {high_risk_cluster}")
    
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    return rfm[['CustomerId', 'is_high_risk']]


class FullDataProcessor:
    """Orchestrates feature engineering and target creation."""
    
    def init(self, reference_date=None, random_state=42):
        self.reference_date = reference_date
        self.random_state = random_state
        self.feature_engineer = FeatureEngineer()
    
    def fit_transform(self, df_raw):
        # 1. Create features
        features = self.feature_engineer.create_customer_features(df_raw)
        # 2. Create target
        target = create_rfm_target(df_raw, self.reference_date, random_state=self.random_state)
        # 3. Merge
        final_df = features.merge(target, on='CustomerId', how='inner')
        # 4. Drop CustomerId (not needed for modeling)
        final_df.drop('CustomerId', axis=1, inplace=True)
        return final_df
