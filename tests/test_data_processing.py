import pytest
import pandas as pd
from src.data_processing import create_high_risk_target, FullDataProcessor

def test_target_creation_returns_0_or_1():
    dummy = pd.DataFrame({
        'CustomerId': [1,1,2,2],
        'Amount': [100,200,50,50],
        'TransactionStartTime': pd.date_range('2025-01-01', periods=4)
    })
    target = create_high_risk_target(dummy, reference_date=pd.Timestamp('2025-01-10'))
    assert target['is_high_risk'].isin([0,1]).all()

def test_full_processor_returns_no_nans():
    dummy = pd.DataFrame({
        'CustomerId': [1,1,2,2],
        'Amount': [100,200,50,150],
        'TransactionStartTime': ['2025-01-01','2025-01-02','2025-01-01','2025-01-03'],
        'ProductCategory': ['A','B','A','A'],
        'ChannelId': ['web','app','web','web']
    })
    proc = FullDataProcessor()
    out = proc.fit_transform(dummy)
    assert out.isnull().sum().sum() == 0
    assert 'is_high_risk' in out.columns

import pytest
import pandas as pd
from src.data_processing import create_rfm_target, FullDataProcessor

def test_rfm_target_binary():
    dummy = pd.DataFrame({
        'CustomerId': [1,1,2,2,3,3],
        'Amount': [100,200,50,50,10,10],
        'TransactionStartTime': pd.date_range('2025-01-01', periods=6)
    })
    target = create_rfm_target(dummy, reference_date=pd.Timestamp('2025-01-10'), random_state=42)
    assert target['is_high_risk'].isin([0,1]).all()
    assert target['is_high_risk'].nunique() == 2

def test_full_processor_no_missing():
    dummy = pd.DataFrame({
        'CustomerId': [1,1,2,2],
        'Amount': [100,200,50,150],
        'TransactionStartTime': ['2025-01-01','2025-01-02','2025-01-01','2025-01-03'],
        'ProductCategory': ['A','B','A','A'],
        'ChannelId': ['web','app','web','web']
    })
    proc = FullDataProcessor(random_state=42)
    out = proc.fit_transform(dummy)
    assert out.isnull().sum().sum() == 0
    assert 'is_high_risk' in out.columns

import pytest
import pandas as pd
from src.data_processing import create_rfm_target, FullDataProcessor

def test_rfm_target_binary():
    """Test that target is binary (0 or 1)."""
    dummy = pd.DataFrame({
        'CustomerId': [1,1,2,2,3,3],
        'Amount': [100,200,50,50,10,10],
        'TransactionStartTime': pd.date_range('2025-01-01', periods=6)
    })
    target = create_rfm_target(dummy, reference_date=pd.Timestamp('2025-01-10'), random_state=42)
    assert target['is_high_risk'].isin([0,1]).all()
    assert target['is_high_risk'].nunique() == 2

def test_full_processor_no_missing():
    """Test that output has no missing values and contains target."""
    dummy = pd.DataFrame({
        'CustomerId': [1,1,2,2],
        'Amount': [100,200,50,150],
        'TransactionStartTime': ['2025-01-01','2025-01-02','2025-01-01','2025-01-03'],
        'ProductCategory': ['A','B','A','A'],
        'ChannelId': ['web','app','web','web']
    })
    proc = FullDataProcessor(random_state=42)
    out = proc.fit_transform(dummy)
    assert out.isnull().sum().sum() == 0
    assert 'is_high_risk' in out.columns

def test_feature_engineer_returns_expected_columns():
    """Test that feature engineer produces at least the mandatory columns."""
    from src.data_processing import FeatureEngineer
    dummy = pd.DataFrame({
        'CustomerId': [1,1,2,2],
        'Amount': [100,200,50,150],
        'TransactionStartTime': ['2025-01-01','2025-01-02','2025-01-01','2025-01-03'],
        'ProductCategory': ['A','B','A','A'],
        'ChannelId': ['web','app','web','web']
    })
    eng = FeatureEngineer()
    features = eng.create_customer_features(dummy)
    assert 'Amount_sum' in features.columns
    assert 'CustomerId' in features.columns
