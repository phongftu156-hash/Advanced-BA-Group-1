"""Protect offline sample reproducibility and avoid persisting network failures."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
clean = importlib.import_module('02_clean')


def listing():
    return pd.DataFrame([dict(city='Hà Nội', district='Quận Đống Đa', ward='Phường X',
                              lat=np.nan, lon=np.nan, geo_precision='')])


@pytest.fixture
def cache_path(tmp_path, monkeypatch):
    monkeypatch.setattr(clean, 'EXTERNAL', tmp_path)
    monkeypatch.setattr(clean.time, 'sleep', lambda _: None)
    return tmp_path / 'geocode_cache.csv'


def test_cached_district_needs_no_network(cache_path, monkeypatch):
    pd.DataFrame([dict(query='Quận Đống Đa, Hà Nội, Vietnam', lat=21.02, lon=105.82)]).to_csv(cache_path, index=False)
    def unexpected(**kwargs):
        pytest.fail('Cached district must work without contacting Nominatim')
    monkeypatch.setattr('geopy.geocoders.Nominatim', unexpected)
    df = listing()
    clean.geocode_missing(df)
    assert df.loc[0, 'geo_precision'] == 'district'
    assert df.loc[0, 'lat'] == 21.02


def test_old_missing_cache_is_retried_and_success_saved(cache_path, monkeypatch):
    q = 'Phường X, Quận Đống Đa, Hà Nội, Vietnam'
    pd.DataFrame([dict(query=q, lat=np.nan, lon=np.nan)]).to_csv(cache_path, index=False)
    calls = []
    def geocode(query):
        calls.append(query)
        return SimpleNamespace(latitude=21.02, longitude=105.82)
    monkeypatch.setattr('geopy.geocoders.Nominatim', lambda **_: SimpleNamespace(geocode=geocode))
    df = listing()
    clean.geocode_missing(df)
    assert calls == [q]
    assert df.loc[0, 'geo_precision'] == 'ward'
    saved = pd.read_csv(cache_path)
    assert len(saved) == 1
    assert saved.loc[0, 'lat'] == 21.02


def test_network_failure_stops_without_poisoning_cache(cache_path, monkeypatch):
    q = 'Phường X, Quận Đống Đa, Hà Nội, Vietnam'
    pd.DataFrame([dict(query=q, lat=np.nan, lon=np.nan)]).to_csv(cache_path, index=False)
    original = cache_path.read_bytes()
    def geocode(query):
        raise TimeoutError('network unavailable')
    monkeypatch.setattr('geopy.geocoders.Nominatim', lambda **_: SimpleNamespace(geocode=geocode))
    df = listing()
    with pytest.raises(RuntimeError, match='Stopping to avoid dropping listings'):
        clean.geocode_missing(df)
    assert cache_path.read_bytes() == original
    assert df.lat.isna().all()


def test_success_is_checkpointed_before_later_network_failure(cache_path, monkeypatch):
    def geocode(query):
        if query.startswith('Phường X,'):
            return SimpleNamespace(latitude=21.02, longitude=105.82)
        raise TimeoutError('network unavailable')
    monkeypatch.setattr('geopy.geocoders.Nominatim', lambda **_: SimpleNamespace(geocode=geocode))
    df = pd.concat([listing(), listing().assign(ward='Phường Y')], ignore_index=True)
    with pytest.raises(RuntimeError):
        clean.geocode_missing(df)
    saved = pd.read_csv(cache_path)
    assert len(saved) == 1
    assert saved.loc[0, 'query'].startswith('Phường X,')
    assert saved[['lat', 'lon']].notna().all().all()
