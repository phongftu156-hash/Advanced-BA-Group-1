"""Phase 1b: backfill photo-upload dates for rows scraped before that field was added.

The first 400 listings were scraped before `img_first_date` / `img_last_date` were
captured (the field was added once the posted-date refresh issue was found). This
re-requests only those detail pages, with the same polite settings as 01, and
rewrites data/raw/mogi_raw.csv in place (same columns, utf-8-sig).
"""
import csv
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mogi_parsers import parse_detail  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "mogi_raw.csv"
UA = "Mozilla/5.0 (compatible; AdvancedBA-ResearchBot/1.0; university course project, non-commercial)"


def worker(urls):
    s = requests.Session()
    s.headers["User-Agent"] = UA
    out = {}
    for u in urls:
        time.sleep(random.uniform(0.8, 1.5))
        try:
            r = s.get(u, timeout=30)
            if r.status_code in (403, 429):
                time.sleep(300)
                continue
            if r.status_code == 200:
                r.encoding = "utf-8"
                d = parse_detail(r.text)
                out[u] = (d["img_first_date"], d["img_last_date"])
        except requests.RequestException:
            pass
    return out


def main():
    df = pd.read_csv(RAW, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    todo = df.loc[df["img_first_date"] == "", "url"].tolist()
    print(f"rows without photo dates: {len(todo)}", flush=True)
    if not todo:
        return
    chunks = [todo[i::3] for i in range(3)]
    res = {}
    with ThreadPoolExecutor(3) as ex:
        for part in ex.map(worker, chunks):
            res.update(part)
    m = df["url"].map(res)
    has = m.notna()
    df.loc[has, "img_first_date"] = m[has].map(lambda x: x[0])
    df.loc[has, "img_last_date"] = m[has].map(lambda x: x[1])
    df.to_csv(RAW, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"backfilled {has.sum()} of {len(todo)} rows", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
