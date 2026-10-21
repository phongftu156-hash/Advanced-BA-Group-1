"""Unit tests for the mogi.vn parsers. Run: python -m pytest code/tests -q

Price strings in REAL_PRICES were copied from data/raw/mogi_raw.csv (Oct 2026 scrape);
the expected values match the numeric JSON-LD price published on the same page.
"""
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mogi_parsers import (floors_from_text, frontage_from_text, keyword_flags,  # noqa: E402
                          listing_id_from_url, normalise_legal, parse_area, parse_dimensions,
                          parse_price, parse_vn_date, road_width_from_text, scrub_personal,
                          split_address)

REAL_PRICES = [
    ("45 tỷ 900 triệu", 45.9e9), ("3 tỷ 950 triệu", 3.95e9), ("10 tỷ 600 triệu", 10.6e9),
    ("20 tỷ", 20e9), ("5 tỷ 600 triệu", 5.6e9), ("17 tỷ 799 triệu", 17.799e9), ("50 tỷ", 50e9),
    ("155 triệu", 155e6), ("13 tỷ 500 triệu", 13.5e9), ("33 tỷ 750 triệu", 33.75e9),
    ("14 tỷ 699 triệu", 14.699e9), ("3 tỷ 650 triệu", 3.65e9), ("50 triệu", 50e6),
    ("22 tỷ 700 triệu", 22.7e9), ("6 tỷ", 6e9), ("495 tỷ", 495e9), ("3 tỷ 581 triệu", 3.581e9),
    ("6 tỷ 50 triệu", 6.05e9), ("7 tỷ 953 triệu", 7.953e9), ("760 tỷ", 760e9),
    ("8 tỷ 950 triệu", 8.95e9), ("15 tỷ 499 triệu", 15.499e9), ("250 triệu", 250e6),
    ("4 tỷ 990 triệu", 4.99e9), ("9 tỷ 900 triệu", 9.9e9),
]


@pytest.mark.parametrize("raw,expected", REAL_PRICES)
def test_real_prices(raw, expected):
    assert parse_price(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw,expected", [
    ("3,2 tỷ", 3.2e9), ("3.2 tỷ", 3.2e9), ("1,5 tỷ", 1.5e9), ("850 triệu", 850e6),
    ("3 Tỷ 200 Triệu", 3.2e9), ("2.500.000.000", 2.5e9),
])
def test_price_formats(raw, expected):
    assert parse_price(raw) == pytest.approx(expected)


def test_price_per_m2_needs_area():
    assert parse_price("45 triệu/m²", area_m2=100) == pytest.approx(4.5e9)
    assert parse_price("45 triệu/m2") is None


@pytest.mark.parametrize("raw", ["Thỏa thuận", "Thoả thuận", "", None, "Liên hệ"])
def test_negotiable(raw):
    assert parse_price(raw) is None


@pytest.mark.parametrize("raw,expected", [
    ("358 m 2", 358), ("146,7 m 2", 146.7), ("97,1 m 2 (5x19)", 97.1), ("90 m2", 90),
    ("0 m 2", None), ("", None), (None, None),
])
def test_area(raw, expected):
    assert parse_area(raw) == (pytest.approx(expected) if expected else None)


def test_dimensions():
    assert parse_dimensions("107 m 2 (5,3x20)") == (5.3, 20.0)
    assert parse_dimensions("52 m 2 (5x10,4)") == (5.0, 10.4)
    assert parse_dimensions("78 m 2") == (None, None)


def test_dates():
    today = date(2026, 10, 15)
    assert parse_vn_date("09/10/2026", today) == date(2026, 10, 9)
    assert parse_vn_date("Hôm nay", today) == today
    assert parse_vn_date("Hôm qua", today) == date(2026, 10, 14)
    assert parse_vn_date("", today) is None


def test_text_features():
    assert floors_from_text("Kết cấu Trệt + 2 lầu") == 3
    assert floors_from_text("Nhà xây 8 tầng cao") == 8
    assert floors_from_text("nhà MP Kim Mã, 20m2x3T") == 3
    assert floors_from_text("dòng tiền 25 tr/ th") is None
    assert road_width_from_text("Nhà hẻm xe hơi 6 m, Lâm Văn Bền") == 6
    assert road_width_from_text("ngõ rộng 3,5m ô tô đỗ cửa") == 3.5
    assert frontage_from_text("MT 5.5M – Giá 20.3 Tỷ") == 5.5
    kw = keyword_flags("Lô góc, ô tô vào nhà, kinh doanh tốt, chính chủ")
    assert kw["kw_corner"] == kw["kw_car_access"] == kw["kw_business"] == kw["kw_owner"] == 1
    assert kw["kw_lake_view"] == 0


def test_legal():
    assert normalise_legal("Sổ hồng") == "Sổ hồng"
    assert normalise_legal("Hợp đồng mua bán") == "Hợp đồng mua bán"
    assert normalise_legal("Đang chờ sổ") == "Đang chờ sổ"
    assert normalise_legal("Giấy tay") == "Khác"
    assert normalise_legal("") == ""


def test_address_and_ids():
    assert split_address("Nguyễn Văn Quá, Phường Đông Hưng Thuận, Quận 12, TPHCM") == \
        ("Nguyễn Văn Quá", "Phường Đông Hưng Thuận", "Quận 12")
    url = "https://mogi.vn/quan-ba-dinh/mua-can-ho-chung-cu/gia-hap-dan-chcc-toa-c7-id22750972"
    assert listing_id_from_url(url) == "22750972"


def test_privacy_scrub():
    s = scrub_personal("Liên hệ: 077 9938 978 - Mr Long, 0855989882, a.b@gmail.com")
    assert "9938" not in s and "0855989882" not in s and "gmail" not in s
