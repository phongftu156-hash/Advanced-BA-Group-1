"""Parsers shared by the mogi.vn scraper (01), cleaning (02) and the tests.

Everything here is pure (no network) so it can be unit-tested on saved strings.
"""
import re
import unicodedata
from datetime import date, datetime

from bs4 import BeautifulSoup

# --------------------------------------------------------------------------- #
# Search cards
# --------------------------------------------------------------------------- #
ID_RE = re.compile(r"-id(\d{6,10})$")


def listing_id_from_url(url):
    m = ID_RE.search(url.split("?")[0])
    return m.group(1) if m else ""


def parse_cards(html):
    """Return the listing cards of one search page."""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for li in soup.select("li:has(> .prop-info)"):
        a = li.select_one("a.link-overlay")
        if not a or not a.get("href"):
            continue
        attrs = [x.get_text("", strip=True) for x in li.select(".prop-attr li")]
        area = next((x for x in attrs if re.search(r"m\s*2?$", x) and "PN" not in x), "")
        bed = next((re.sub(r"\D", "", x) for x in attrs if x.endswith("PN")), "")
        bath = next((re.sub(r"\D", "", x) for x in attrs if x.endswith("WC")), "")
        price = li.select_one(".price")
        created = li.select_one(".prop-created")
        nimg = li.select_one(".prop-img .total span")
        title = li.select_one(".prop-title")
        addr = li.select_one(".prop-addr")
        out.append({
            "listing_id": listing_id_from_url(a["href"]),
            "url": a["href"].split("?")[0],
            "card_title": title.get_text(" ", strip=True) if title else "",
            "card_price": price.get_text(" ", strip=True) if price else "",
            "card_area": area,
            "card_bedrooms": bed,
            "card_bathrooms": bath,
            "card_addr": addr.get_text(" ", strip=True) if addr else "",
            "card_date": created.get_text(" ", strip=True) if created else "",
            "card_n_images": nimg.get_text(strip=True) if nimg else "",
        })
    return out


# --------------------------------------------------------------------------- #
# Detail pages
# --------------------------------------------------------------------------- #
PHONE_RE = re.compile(r"(?:\+?84|0)(?:[\s\.\-]?\d){8,10}")
EMAIL_RE = re.compile(r"[\w\.\-]+@[\w\-]+\.[\w\.]+")
WARD_RE = re.compile(r"^(Phường|Xã|Thị trấn|P\.)\s*", re.I)
DISTRICT_RE = re.compile(r"^(Quận|Huyện|Thị xã|Thành phố|TP\.)\s*", re.I)
CITY_NAMES = {"Hà Nội": "Hà Nội", "TPHCM": "TP. Hồ Chí Minh", "Hồ Chí Minh": "TP. Hồ Chí Minh",
              "TP. Hồ Chí Minh": "TP. Hồ Chí Minh", "TP.HCM": "TP. Hồ Chí Minh"}


def scrub_personal(text):
    """Remove phone numbers and e-mail addresses (no personal data is stored)."""
    text = PHONE_RE.sub("[SĐT]", text or "")
    return EMAIL_RE.sub("[email]", text)


def parse_detail(html):
    """Parse one listing detail page into a flat dict of raw fields."""
    soup = BeautifulSoup(html, "lxml")
    d = {}
    h1 = soup.select_one(".main-info .title h1") or soup.select_one("h1")
    d["title"] = h1.get_text(" ", strip=True) if h1 else ""
    addr = soup.select_one(".main-info .address") or soup.select_one(".address")
    d["address_raw"] = addr.get_text(" ", strip=True) if addr else ""
    pr = soup.select_one(".main-info .price")
    d["price_raw"] = pr.get_text(" ", strip=True) if pr else ""

    attrs = {}
    for at in soup.select(".info-attr"):
        t = at.get_text("|", strip=True).split("|")
        if t:
            attrs[t[0].strip()] = " ".join(x.strip() for x in t[1:]).strip()
    d["area_used_raw"] = attrs.get("Diện tích sử dụng", "")
    d["land_area_raw"] = attrs.get("Diện tích đất", "")
    d["bedrooms"] = re.sub(r"\D", "", attrs.get("Phòng ngủ", ""))
    d["bathrooms"] = re.sub(r"\D", "", attrs.get("Nhà tắm", ""))
    d["legal_raw"] = attrs.get("Pháp lý", "")
    d["posted_date"] = attrs.get("Ngày đăng", "")
    d["updated_date"] = attrs.get("Ngày cập nhật", "")
    d["site_code"] = attrs.get("Mã BĐS", "")
    known = {"Diện tích sử dụng", "Diện tích đất", "Phòng ngủ", "Nhà tắm", "Pháp lý",
             "Ngày đăng", "Ngày cập nhật", "Mã BĐS"}
    d["other_attrs"] = "; ".join(f"{k}={v}" for k, v in attrs.items() if k not in known)

    # breadcrumb: Mogi > Mua bán nhà đất > City > District > "Bán <type> <street>"
    bc = [x.get_text(" ", strip=True) for x in soup.select(".breadcrumb li")]
    d["bc_city"] = bc[2] if len(bc) > 2 else ""
    d["bc_district"] = bc[3] if len(bc) > 3 else ""
    d["bc_last"] = bc[4] if len(bc) > 4 else ""

    # exact coordinates from the embedded Google Maps iframe (q=lat,lon)
    m = re.search(r"maps/embed/v1/place\?[^\"']*?q=(-?\d+\.\d+),(-?\d+\.\d+)", html)
    d["lat"], d["lon"] = (m.group(1), m.group(2)) if m else ("", "")

    # JSON-LD offer: numeric price and schema type
    m = re.search(r'"priceCurrency":"VND","price":(\d+)', html)
    d["price_vnd_ld"] = m.group(1) if m else ""
    m = re.search(r'"itemOffered":\{"@type":"(\w+)"', html)
    d["schema_type"] = m.group(1) if m else ""

    body = soup.select_one(".info-content-body")
    desc = body.get_text("\n", strip=True) if body else ""
    d["description_full"] = scrub_personal(desc)  # used for text features, then truncated

    # seller: registered broker profile (/moi-gioi/ link) vs plain account; never store names/phones
    agent = soup.select_one(".agent-info .agent-name a")
    d["seller_profile"] = "broker_profile" if agent and "/moi-gioi/" in agent.get("href", "") else "no_profile"
    tenure = soup.select_one(".agent-info .agent-date")
    d["agent_tenure_months"] = tenure_months(tenure.get_text(" ", strip=True)) if tenure else ""

    imgs = set(re.findall(r"cloud\.mogi\.vn/images/(?:thumb-[a-z]+/)?(\d{4}/\d\d/\d\d/\d+/[0-9a-f]+)\.jpg", html))
    d["n_images"] = len(imgs)
    # photo upload dates (from the image storage path) proxy the listing creation date
    days = sorted(i[:10] for i in imgs)
    d["img_first_date"] = days[0].replace("/", "-") if days else ""
    d["img_last_date"] = days[-1].replace("/", "-") if days else ""
    return d


def tenure_months(text):
    y = re.search(r"(\d+)\s*năm", text)
    mo = re.search(r"(\d+)\s*tháng", text)
    if not y and not mo:
        return ""
    return (int(y.group(1)) if y else 0) * 12 + (int(mo.group(1)) if mo else 0)


def split_address(address_raw):
    """'Street, Phường X, Quận Y, City' -> (street, ward, district)."""
    street = ward = district = ""
    parts = [p.strip() for p in (address_raw or "").split(",") if p.strip()]
    for p in parts:
        if WARD_RE.match(p) and not ward:
            ward = p
        elif DISTRICT_RE.match(p) and not district:
            district = p
    if parts and not WARD_RE.match(parts[0]) and not DISTRICT_RE.match(parts[0]) \
            and parts[0] not in CITY_NAMES:
        street = parts[0]
    return street, ward, district


def normalise_city(name):
    return CITY_NAMES.get((name or "").strip(), (name or "").strip())


# --------------------------------------------------------------------------- #
# Values
# --------------------------------------------------------------------------- #
def _num(s):
    """'3,2' or '3.2' -> 3.2 ; '1.200' (thousands) -> 1200."""
    s = s.strip()
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        return float(s.replace(".", ""))
    return float(s.replace(",", "."))


def parse_price(raw, area_m2=None):
    """Vietnamese asking price string -> VND (float) or None.

    Handles 'tỷ', 'triệu', 'nghìn', mixed forms ('3 tỷ 200 triệu'), decimal commas
    ('3,2 tỷ') and per-m2 prices ('45 triệu/m²', needs area). Negotiable -> None.
    """
    if raw is None:
        return None
    s = unicodedata.normalize("NFC", str(raw)).lower().strip()
    if not s or "thỏa thuận" in s or "thoả thuận" in s or "liên hệ" in s:
        return None
    per_m2 = bool(re.search(r"/\s*m", s))
    total = 0.0
    found = False
    for val, unit in re.findall(r"(\d+(?:[.,]\d+)*)\s*(tỷ|ty|triệu|tr|nghìn|ngàn|k)\b", s):
        v = _num(val)
        mult = {"tỷ": 1e9, "ty": 1e9, "triệu": 1e6, "tr": 1e6, "nghìn": 1e3, "ngàn": 1e3, "k": 1e3}[unit]
        total += v * mult
        found = True
    if not found:
        m = re.fullmatch(r"(\d[\d\.,]*)\s*(đ|vnd|vnđ)?", s)
        if not m:
            return None
        total = float(re.sub(r"[^\d]", "", m.group(1)))
        if total <= 0:
            return None
    if per_m2:
        if not area_m2:
            return None
        total *= float(area_m2)
    return total if total > 0 else None


def parse_area(raw):
    """'96 m2 (4,3x23)' / '246,4 m 2' -> 96.0 / 246.4 ; None if missing or zero."""
    if raw is None:
        return None
    s = unicodedata.normalize("NFC", str(raw)).strip()
    m = re.match(r"\s*(\d+(?:[.,]\d+)*)\s*m", s)
    if not m:
        return None
    v = _num(m.group(1))
    return v if v > 0 else None


def parse_dimensions(raw):
    """'96 m2 (4,3x23)' -> (4.3, 23.0) as (width/frontage, depth)."""
    m = re.search(r"\((\d+(?:[.,]\d+)?)\s*[x×]\s*(\d+(?:[.,]\d+)?)\)", str(raw or ""))
    if not m:
        return None, None
    return _num(m.group(1)), _num(m.group(2))


def parse_vn_date(raw, today=None):
    """'15/10/2026' -> date ; 'Hôm nay' / 'Hôm qua' handled relative to today."""
    s = (raw or "").strip()
    today = today or date.today()
    if not s:
        return None
    if s.lower() == "hôm nay":
        return today
    if s.lower() == "hôm qua":
        return date.fromordinal(today.toordinal() - 1)
    try:
        return datetime.strptime(s, "%d/%m/%Y").date()
    except ValueError:
        return None


LEGAL_MAP = [
    (r"sổ hồng", "Sổ hồng"),
    (r"sổ đỏ", "Sổ đỏ"),
    (r"hợp đồng mua bán|hđmb|hdmb", "Hợp đồng mua bán"),
    (r"chờ sổ|đang chờ|đợi sổ|đang làm sổ|sắp có sổ", "Đang chờ sổ"),
]


def normalise_legal(raw):
    s = (raw or "").lower()
    if not s.strip():
        return ""
    for pat, lab in LEGAL_MAP:
        if re.search(pat, s):
            return lab
    return "Khác"


# --------------------------------------------------------------------------- #
# Text-derived features (title + description)
# --------------------------------------------------------------------------- #
def floors_from_text(text):
    """Number of storeys. 'trệt + 2 lầu' -> 3 ; '5 tầng' -> 5 ; '4T' -> 4."""
    s = (text or "").lower()
    lau = re.search(r"(\d{1,2})\s*lầu", s)
    tang = re.search(r"(\d{1,2})\s*tầng(?!\s*(?:hầm|\d))", s)
    short = re.search(r"m2?\s*[x×]\s*(\d{1,2})\s*t\b", s)  # Hanoi shorthand '56m2x8T'
    cands = []
    if lau:  # southern usage: 'n lầu' counts storeys above the ground floor (trệt)
        cands.append(int(lau.group(1)) + 1)
    if tang:
        cands.append(int(tang.group(1)))
    if short and not cands:
        cands.append(int(short.group(1)))
    cands = [c for c in cands if 1 <= c <= 30]
    return max(cands) if cands else None


ROAD_RE = re.compile(
    r"(?:hẻm|hẽm|ngõ|ngách|kiệt|đường|lộ giới|đường trước nhà|mặt đường|đường vào)\s*"
    r"(?:xe hơi\s*|ô tô\s*|oto\s*|nhựa\s*|bê tông\s*|trước nhà\s*)?(?:rộng\s*|thông\s*)?"
    r"(\d{1,2}(?:[.,]\d)?)\s*m(?![²2\w])")


def road_width_from_text(text):
    s = (text or "").lower()
    vals = [_num(v) for v in ROAD_RE.findall(s)]
    vals = [v for v in vals if 0.8 <= v <= 60]
    return max(vals) if vals else None


FRONT_RE = re.compile(r"(?:mặt tiền|mt|ngang)\s*(?:rộng\s*)?(\d{1,2}(?:[.,]\d{1,2})?)\s*m(?![²2\w])")


def frontage_from_text(text):
    s = (text or "").lower()
    vals = [_num(v) for v in FRONT_RE.findall(s)]
    vals = [v for v in vals if 1 <= v <= 100]
    return vals[0] if vals else None


KEYWORDS = {
    "kw_car_access": r"ô\s*tô|oto|xe hơi|ô-tô",
    "kw_corner": r"\bgóc\b|lô góc|căn góc|2 mặt tiền|hai mặt tiền",
    "kw_business": r"kinh doanh|buôn bán|cho thuê",
    "kw_lake_view": r"view hồ|nhìn hồ|view sông|mặt hồ|ven hồ",
    "kw_near_school": r"gần trường|cạnh trường|trường học",
    "kw_owner": r"chính chủ",
    "kw_urgent": r"bán gấp|cần bán gấp|cắt lỗ|giảm giá|hạ giá|ngộp",
    "kw_furnished": r"full nội thất|đầy đủ nội thất|nội thất cao cấp|nội thất đầy đủ|full đồ",
}


def keyword_flags(text):
    s = (text or "").lower()
    return {k: int(bool(re.search(p, s))) for k, p in KEYWORDS.items()}


PROJECT_RE = re.compile(
    r"\b(vinhomes?\s+[\w\s]{2,25}?|masteri\s+[\w\s]{2,20}?|chung cư\s+[A-ZĐ][\w\s\-]{2,30}?|"
    r"dự án\s+[A-ZĐ][\w\s\-]{2,30}?|khu đô thị\s+[A-ZĐ][\w\s\-]{2,30}?|kđt\s+[A-ZĐ][\w\s\-]{2,30}?)"
    r"(?=[,\.\-\|\(]|\s+(?:giá|bán|căn|tầng|tòa|view|full|\d)|$)", re.I)


def project_from_text(title):
    m = PROJECT_RE.search(title or "")
    return m.group(1).strip() if m else ""
