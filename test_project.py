import pandas as pd
import numpy as np
import pytest
from project import load_data, compute_stats, compare_conditions

def test_load_data():
    df = load_data('EST_Tae_CCR_Rawdata_Figure1.csv')
    assert list(df.columns) == ['time_days', 'concentration', 'replicate', 'temperature']
    assert len(df) > 0
    assert set(df['temperature'].unique()) == {'50C', '30C', '2C'}

def test_compute_stats():
    df = load_data('EST_Tae_CCR_Rawdata_Figure1.csv')
    stats = compute_stats(df)
    assert 'mean' in stats.columns
    assert 'std' in stats.columns
    assert all(stats['mean'] >= 0)

def test_compare_conditions():
    df = load_data('EST_Tae_CCR_Rawdata_Figure1.csv')
    result = compare_conditions(df, '50C', '30C', 30.0)
    assert 'p_value' in result
    assert 'significant' in result
    assert isinstance(result['significant'], (bool, np.bool_))