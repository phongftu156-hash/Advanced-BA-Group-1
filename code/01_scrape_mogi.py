"""Phase 1: scrape mogi.vn sale listings (Hanoi + HCMC, all property types).

Extends the 300-listing test scraper (code/scraper.py): same card selectors and
?cp=N pagination, now with quotas per city x property group, an 80/20 split
between newest-first pages and the oldest pages (the site has no "oldest" sort,
so the tail of the newest-first list is paged backwards), a detail-page request
for every listing, 3 concurrent polite workers with adaptive throttling,
retries/backoff, checkpoints every 100 rows and a resumable state file.

Stops at 6,000 raw listings or after 1h45m of scraping, whichever comes first.

Usage:  python code/01_scrape_mogi.py            (resumes if state exists)
        python code/01_scrape_mogi.py --fresh    (ignores existing state)
"""
import argparse
import csv
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mogi_parsers import (floors_from_text, frontage_from_text, keyword_flags,  # noqa: E402
                          normalise_city, normalise_legal, parse_area, parse_cards,
                          parse_detail, parse_dimensions, parse_price, project_from_text,
                          road_width_from_text, split_address)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "mogi_raw.csv"
STATE = ROOT / "data" / "raw" / "scrape_state.json"
LOG = ROOT / "result" / "logs" / "scrape_log.txt"
BASE = "https://mogi.vn"
UA = "Mozilla/5.0 (compatible; AdvancedBA-ResearchBot/1.0; university course project, non-commercial)"
N_WORKERS = 3
DELAY_FAST = (0.8, 1.5)      # per worker, normal mode
DELAY_SLOW = (2.0, 3.0)      # global spacing after a block (1 worker)
TARGET_RAW = 6_000
TIME_BUDGET_S = 105 * 60     # 1h45m
MIN_ELIGIBLE = 3_000         # minimum acceptable sample (retry rule)
PER_PAGE = 15

CITIES = {"ha-noi": "Hà Nội", "ho-chi-minh": "TP. Hồ Chí Minh"}
# property group -> (quota per city, [(search slug, land_subtype)])
GROUPS = {
    "Căn hộ chung cư": (900, [("mua-can-ho-chung-cu", "")]),
    "Nhà hẻm/ngõ": (800, [("mua-nha-hem-ngo", ""), ("mua-duong-noi-bo", "")]),
    "Nhà mặt tiền/phố": (500, [("mua-nha-mat-tien-pho", "")]),
    "Biệt thự/liền kề": (350, [("mua-nha-biet-thu-lien-ke", "")]),
    "Đất": (450, [("mua-dat-tho-cu", "Đất thổ cư"), ("mua-dat-nen-du-an", "Đất nền dự án")]),
}
SLUG_GROUP = {slug: (g, sub) for g, (_, subs) in GROUPS.items() for slug, sub in subs}

FIELDS = ["listing_id", "url", "city", "district", "ward", "street", "address_raw",
          "property_type", "subcategory", "land_subtype", "price_raw", "price_vnd",
          "price_vnd_ld", "area_m2", "area_used_raw", "land_area_raw", "price_per_m2_raw",
          "bedrooms", "bathrooms", "floors", "floors_from_text", "frontage_m",
          "frontage_from_text", "depth_m", "road_width_m", "road_width_from_text",
          "legal_raw", "legal_status", "legal_from_text", "furnishing", "project_name",
          "posted_date", "updated_date", "card_date", "seller_type", "agent_tenure_months",
          "n_images", "title", "description", "description_len", "lat", "lon",
          "geo_precision", "schema_type", "other_attrs", "kw_car_access", "kw_corner",
          "kw_business", "kw_lake_view", "kw_near_school", "kw_owner", "kw_urgent",
          "kw_furnished", "sample_part", "search_page", "scraped_at", "img_first_date",
          "img_last_date"]

_log_lock = threading.Lock()


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    with _log_lock:
        print(line, flush=True)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class Blocked(Exception):
    pass


class Throttle:
    """Adaptive concurrency gate shared by all workers.

    Normal: up to 3 concurrent requests, each worker waits 0.8-1.5 s between its own
    requests. First 403/429: drop to 1 worker, 2-3 s global spacing, pause 5 min.
    Further blocks while degraded: pause 15 min, then 30 min, then give up.
    15 min without a block while degraded: go back to 2 workers (0.8-1.5 s).
    """

    def __init__(self):
        self.cv = threading.Condition()
        self.allowed = N_WORKERS
        self.active = 0
        self.pause_until = 0.0
        self.last_block = None
        self.last_global = 0.0
        self.escalation = [300, 900, 1800]
        self.n_blocks = 0
        self.local = threading.local()

    def _slow(self):
        return self.allowed == 1

    def acquire(self):
        with self.cv:
            while True:
                now = time.time()
                if (self.last_block and self.allowed == 1 and now - self.last_block >= 900
                        and now >= self.pause_until):
                    self.allowed = 2
                    log("THROTTLE: 15 min without a block -> 2 workers, 0.8-1.5 s delay")
                    self.cv.notify_all()
                if now < self.pause_until:
                    self.cv.wait(min(self.pause_until - now, 30))
                    continue
                if self.active >= self.allowed:
                    self.cv.wait(5)
                    continue
                if self._slow():
                    gap = random.uniform(*DELAY_SLOW) - (now - self.last_global)
                    if gap > 0:
                        self.cv.wait(gap)
                        continue
                self.active += 1
                self.last_global = time.time()
                return

    def release(self):
        with self.cv:
            self.active -= 1
            self.cv.notify_all()

    def blocked(self, code):
        with self.cv:
            now = time.time()
            if now < self.pause_until:  # another worker already reacted to this block
                return
            if not self.escalation:
                raise Blocked(f"HTTP {code} after all pauses")
            p = self.escalation.pop(0)
            self.n_blocks += 1
            self.allowed = 1
            self.last_block = now
            self.pause_until = now + p
            log(f"THROTTLE: HTTP {code} -> 1 worker, 2-3 s delay, pause {p // 60} min")

    def session(self):
        s = getattr(self.local, "s", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"User-Agent": UA, "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.5"})
            self.local.s = s
            self.local.last = 0.0
        return s

    def worker_delay(self):
        if not self._slow():
            wait = random.uniform(*DELAY_FAST) - (time.time() - self.local.last)
            if wait > 0:
                time.sleep(wait)


TH = Throttle()
STATS = {"req": 0, "fail": 0}
_stats_lock = threading.Lock()


def fetch(url):
    """GET with per-worker delay, 3 retries on timeouts/5xx, adaptive throttle on 403/429."""
    sess = TH.session()
    attempt = 0
    while True:
        TH.worker_delay()
        TH.acquire()
        try:
            TH.local.last = time.time()
            r = sess.get(url, timeout=30)
        except requests.RequestException as e:
            r, err = None, e
        finally:
            TH.release()
        with _stats_lock:
            STATS["req"] += 1
        if r is None:
            attempt += 1
            if attempt >= 3:
                log(f"  [fail] {url}: {err}")
                with _stats_lock:
                    STATS["fail"] += 1
                return None
            time.sleep(5 * 2 ** attempt)
            continue
        if r.status_code in (403, 429):
            TH.blocked(r.status_code)
            continue
        if r.status_code >= 500:
            attempt += 1
            if attempt >= 3:
                log(f"  [fail] {r.status_code} {url}")
                with _stats_lock:
                    STATS["fail"] += 1
                return None
            time.sleep(5 * 2 ** attempt)
            continue
        if r.status_code != 200:
            with _stats_lock:
                STATS["fail"] += 1
            return None
        r.encoding = "utf-8"
        return r.text


def has_cards(url):
    html = fetch(url)
    return bool(html and parse_cards(html))


def page_count(slug_url):
    """Last search page that still returns cards (binary search)."""
    lo, hi = 1, 400
    if has_cards(f"{slug_url}?cp={hi}"):
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if has_cards(f"{slug_url}?cp={mid}"):
            lo = mid
        else:
            hi = mid
    return lo


def allocate(quotas, supply):
    """Cap quotas by supply; reallocate shortfall proportionally to groups with spare supply."""
    alloc = {k: min(quotas[k], supply[k]) for k in quotas}
    for _ in range(10):
        short = sum(quotas.values()) - sum(alloc.values())
        spare = {k: supply[k] - alloc[k] for k in quotas if supply[k] > alloc[k]}
        if short <= 0 or not spare:
            break
        wsum = sum(quotas[k] for k in spare)
        for k in spare:
            alloc[k] = min(supply[k], alloc[k] + round(short * quotas[k] / wsum))
    return alloc


def build_plan(pool):
    jobs = [(cs, slug) for cs in CITIES for _, (_, subs) in GROUPS.items() for slug, _ in subs]
    pages = dict(zip(jobs, pool.map(lambda j: page_count(f"{BASE}/{j[0]}/{j[1]}"), jobs)))
    plan = []
    for cs, cname in CITIES.items():
        sub_supply, g_supply = {}, {}
        for g, (_, subs) in GROUPS.items():
            for slug, _ in subs:
                sub_supply[slug] = pages[(cs, slug)] * PER_PAGE
                log(f"supply {cname} / {slug}: {pages[(cs, slug)]} pages "
                    f"(~{sub_supply[slug]} listings)")
            g_supply[g] = sum(sub_supply[s] for s, _ in subs)
        g_alloc = allocate({g: q for g, (q, _) in GROUPS.items()}, g_supply)
        for g, (_, subs) in GROUPS.items():
            s_alloc = allocate({s: max(1, round(g_alloc[g] * sub_supply[s] / max(g_supply[g], 1)))
                                for s, _ in subs}, {s: sub_supply[s] for s, _ in subs})
            for slug, _ in subs:
                n, sup = s_alloc[slug], sub_supply[slug]
                last = sup // PER_PAGE
                full = n >= 0.95 * sup
                plan.append({
                    "city_slug": cs, "city": cname, "group": g, "slug": slug,
                    "alloc": n, "supply": sup, "last_page": last,
                    "target_new": n if full else round(0.8 * n),
                    "target_old": 0 if full else n - round(0.8 * n),
                    "next_new": 1, "next_old": last, "got_new": 0, "got_old": 0, "done": False,
                })
                log(f"plan {cname} / {g} / {slug}: alloc {n} of ~{sup} "
                    f"({'all pages' if full else '80% newest + 20% oldest pages'})")
    return plan


def to_row(card, det, task, part, page):
    text = f"{det.get('title', '')}\n{det.get('description_full', '')}"
    street, ward, district = split_address(det.get("address_raw", ""))
    district = det.get("bc_district") or district
    ptype = task["group"]
    area_used = parse_area(det.get("area_used_raw")) or parse_area(card.get("card_area"))
    area_land = parse_area(det.get("land_area_raw"))
    # apartments: usable floor area; houses, villas, land: land plot area (VN market convention)
    area = area_used if ptype == "Căn hộ chung cư" else (area_land or area_used)
    width, depth = parse_dimensions(det.get("land_area_raw"))
    f_text = frontage_from_text(text)
    floors = floors_from_text(text) if ptype not in ("Căn hộ chung cư", "Đất") else None
    road = road_width_from_text(text)
    legal = normalise_legal(det.get("legal_raw"))
    legal_txt = 0
    if not legal:
        legal = normalise_legal(text) if normalise_legal(text) != "Khác" else ""
        legal_txt = int(bool(legal))
    kw = keyword_flags(text)
    desc = det.get("description_full", "")
    row = {
        "listing_id": card["listing_id"], "url": card["url"],
        "city": normalise_city(det.get("bc_city")) or task["city"],
        "district": district, "ward": ward, "street": street,
        "address_raw": det.get("address_raw", ""),
        "property_type": ptype, "subcategory": task["slug"],
        "land_subtype": SLUG_GROUP[task["slug"]][1],
        "price_raw": det.get("price_raw") or card.get("card_price", ""),
        "price_vnd": parse_price(det.get("price_raw") or card.get("card_price"), area) or "",
        "price_vnd_ld": det.get("price_vnd_ld", ""),
        "area_m2": area or "", "area_used_raw": det.get("area_used_raw", ""),
        "land_area_raw": det.get("land_area_raw", ""), "price_per_m2_raw": "",
        "bedrooms": det.get("bedrooms") or card.get("card_bedrooms", ""),
        "bathrooms": det.get("bathrooms") or card.get("card_bathrooms", ""),
        "floors": floors or "", "floors_from_text": int(floors is not None),
        "frontage_m": width or f_text or "",
        "frontage_from_text": int(not width and f_text is not None),
        "depth_m": depth or "",
        "road_width_m": road or "", "road_width_from_text": int(road is not None),
        "legal_raw": det.get("legal_raw", ""), "legal_status": legal, "legal_from_text": legal_txt,
        "furnishing": "Đầy đủ nội thất" if kw["kw_furnished"] else "",
        "project_name": project_from_text(det.get("title", "")),
        "posted_date": det.get("posted_date", ""), "updated_date": det.get("updated_date", ""),
        "card_date": card.get("card_date", ""),
        "seller_type": det.get("seller_profile", ""),
        "agent_tenure_months": det.get("agent_tenure_months", ""),
        "n_images": det.get("n_images") or card.get("card_n_images", ""),
        "title": det.get("title") or card.get("card_title", ""),
        "description": desc[:1000], "description_len": len(desc),
        "lat": det.get("lat", ""), "lon": det.get("lon", ""),
        "geo_precision": "exact" if det.get("lat") else "",
        "schema_type": det.get("schema_type", ""), "other_attrs": det.get("other_attrs", ""),
        "sample_part": part, "search_page": page,
        "img_first_date": det.get("img_first_date", ""), "img_last_date": det.get("img_last_date", ""),
        "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    row.update(kw)
    return row


def eligible(r):
    return bool(r["price_vnd"]) and bool(r["area_m2"]) and bool(r["district"])


def append_rows(rows):
    new = not RAW.exists()
    with open(RAW, "a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def save_state(state):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE)


def progress(state, t0):
    by = {}
    for t in state["plan"]:
        k = f"{t['city']} | {t['group']}"
        by[k] = by.get(k, 0) + t["got_new"] + t["got_old"]
    total = state["n_rows"]
    target = min(TARGET_RAW, sum(t["alloc"] for t in state["plan"]))
    el = state["elapsed_s"] + time.time() - t0
    rate = total / el * 3600 if el > 0 else 0
    eta = (target - total) / rate if rate > 0 else float("nan")
    log(f"PROGRESS rows={total} eligible={state['n_eligible']} requests={STATS['req']} "
        f"fail_rate={STATS['fail'] / max(STATS['req'], 1):.1%} blocks={TH.n_blocks} "
        f"elapsed={el / 3600:.2f}h throughput={rate:.0f} listings/h "
        f"projected_remaining={eta:.2f}h")
    for k in sorted(by):
        log(f"    {k}: {by[k]}")


def next_page(t):
    """Pick the next search page for a task, or mark it done."""
    if t["got_new"] < t["target_new"] and t["next_new"] <= t["last_page"] + 3:
        t["next_new"] += 1
        return "newest", t["next_new"] - 1
    if t["got_old"] < t["target_old"] and t["next_old"] >= t["next_new"]:
        t["next_old"] -= 1
        return "oldest", t["next_old"] + 1
    t["done"] = True
    return None, None


def run(fresh=False):
    pool = ThreadPoolExecutor(max_workers=N_WORKERS)
    t0 = time.time()
    if STATE.exists() and not fresh:
        state = json.loads(STATE.read_text(encoding="utf-8"))
        log(f"resuming: {state['n_rows']} rows already saved, "
            f"{state['elapsed_s'] / 60:.0f} min already used")
    else:
        log("robots.txt (checked 2026-10-15): Allow / ; Disallow /trang-ca-nhan/, /api/, "
            "/template/, /MarketPrice/, /Property/, *gclid=, *wbraid= -> search and detail "
            "pages used here are allowed")
        log(f"User-Agent: {UA}; {N_WORKERS} workers, {DELAY_FAST[0]}-{DELAY_FAST[1]} s per worker; "
            f"target {TARGET_RAW} raw or {TIME_BUDGET_S // 60} min")
        state = {"plan": build_plan(pool), "visited": [], "failed": [], "n_rows": 0,
                 "n_eligible": 0, "elapsed_s": 0.0}
        save_state(state)
    visited = set(state["visited"])
    buf = []
    next_log = (state["n_rows"] // 500 + 1) * 500
    stop_reason = None

    def elapsed():
        return state["elapsed_s"] + time.time() - t0

    def flush():
        nonlocal buf, t0
        if buf:
            append_rows(buf)
            buf = []
        state["visited"] = sorted(visited)
        now = time.time()
        state["elapsed_s"] += now - t0
        t0 = now
        save_state(state)

    try:
        while True:
            if state["n_rows"] >= TARGET_RAW:
                stop_reason = f"reached {TARGET_RAW} raw listings"
                break
            if elapsed() >= TIME_BUDGET_S:
                stop_reason = "time budget (1h45m) reached"
                break
            # one round = one search page from every active task, fetched in parallel
            picks = []
            for t in state["plan"]:
                if not t["done"]:
                    part, page = next_page(t)
                    if part:
                        picks.append((t, part, page))
            if not picks:
                stop_reason = "all category allocations exhausted"
                break
            urls = [f"{BASE}/{t['city_slug']}/{t['slug']}" + (f"?cp={p}" if p > 1 else "")
                    for t, _, p in picks]
            pages = list(pool.map(fetch, urls))
            jobs = []
            for (t, part, page), html in zip(picks, pages):
                cards = parse_cards(html) if html else []
                if not cards and part == "newest":
                    t["target_new"] = t["got_new"]  # list ended earlier than estimated
                need = (t["target_new"] - t["got_new"]) if part == "newest" else (t["target_old"] - t["got_old"])
                for c in cards:
                    if need <= 0 or state["n_rows"] + len(jobs) >= TARGET_RAW:
                        break
                    if not c["listing_id"] or c["listing_id"] in visited:
                        continue
                    if c["url"].split("/")[4:5] != [t["slug"]]:
                        continue  # card from another category (promoted listing)
                    visited.add(c["listing_id"])
                    jobs.append((t, part, page, c))
                    need -= 1
            details = list(pool.map(lambda j: fetch(j[3]["url"]), jobs))
            for (t, part, page, c), dh in zip(jobs, details):
                if dh is None:
                    state["failed"].append(c["listing_id"])
                    continue
                try:
                    row = to_row(c, parse_detail(dh), t, part, page)
                except Exception as e:  # parsing problem: skip listing, keep going
                    state["failed"].append(c["listing_id"])
                    log(f"  [parse error] {c['url']}: {e}")
                    continue
                buf.append(row)
                t["got_new" if part == "newest" else "got_old"] += 1
                state["n_rows"] += 1
                state["n_eligible"] += eligible(row)
                if len(buf) >= 100:
                    flush()
                if state["n_rows"] >= next_log:
                    progress(state, t0)
                    next_log += 500
    except Blocked as e:
        flush()
        log(f"BLOCKED after all pauses ({e}); stopping with {state['n_rows']} rows")
        state["blocked"] = True
        save_state(state)
        pool.shutdown(wait=False, cancel_futures=True)
        return state
    flush()
    pool.shutdown()
    progress(state, time.time())
    log(f"DONE ({stop_reason}) rows={state['n_rows']} eligible={state['n_eligible']} "
        f"failed={len(state['failed'])} elapsed={state['elapsed_s'] / 60:.1f} min "
        f"throughput={state['n_rows'] / max(state['elapsed_s'], 1) * 3600:.0f} listings/h")
    state["finished"] = True
    state["stop_reason"] = stop_reason
    save_state(state)
    return state


def main():
    global TH
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    if args.fresh:
        for p in (RAW, STATE):
            if p.exists():
                p.unlink()
    state = run(args.fresh)
    # blocked below the minimum acceptable sample -> retry every 30 min for up to 4 h
    tries = 0
    while state.get("blocked") and state["n_eligible"] < MIN_ELIGIBLE and tries < 8:
        tries += 1
        log(f"usable rows < {MIN_ELIGIBLE}: waiting 30 min before retrying")
        time.sleep(1800)
        TH = Throttle()
        state["blocked"] = False
        state["elapsed_s"] = min(state["elapsed_s"], TIME_BUDGET_S - 1800)
        save_state(state)
        state = run(False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
