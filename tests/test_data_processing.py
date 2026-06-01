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
