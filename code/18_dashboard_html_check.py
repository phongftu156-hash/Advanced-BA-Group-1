"""Headless test of the standalone HTML dashboard.

Opens report/dashboard/vietnam_avm_dashboard.html with Playwright (Chromium), fills the form for three example
properties, compares fair value, range, price gap, stale-risk score and comparables with the Python app logic,
and saves screenshots at 1440 px and 390 px wide to result/figures/dashboard_html_*.png.
Report: result/logs/dashboard_html_check.txt.
"""
import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIG, ROOT  # noqa: E402

exp = importlib.import_module("17_export_dashboard")
HTML = ROOT / "report" / "vietnam_avm_dashboard.html"
CASES = [
    {"name": "apartment_caugiay", "city": "Hà Nội", "district": "Quận Cầu Giấy", "type": "Căn hộ chung cư", "ask": 5.5},
    {"name": "alley_dongda", "city": "Hà Nội", "district": "Quận Đống Đa", "type": "Nhà hẻm/ngõ", "ask": 9.5},
    {"name": "land_thuduc", "city": "TP. Hồ Chí Minh", "district": "Quận 9 (TP. Thủ Đức)", "type": "Đất", "ask": 8.0},
]


def main():
    from playwright.sync_api import sync_playwright
    df, avm, conf, stale, seg, lookup, scores = exp.load()
    med = df.groupby("property_type")[["area_m2", "bedrooms", "bathrooms", "floors", "frontage_m",
                                        "road_width_m"]].median()
    lines, ok_all = ["Standalone HTML dashboard check", f"file: {HTML.relative_to(ROOT)} "
                     f"({HTML.stat().st_size / 1e6:.1f} MB)", ""], True
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        for width, height, tag in ((1440, 1000, "desktop"), (390, 844, "phone")):
            pg = b.new_page(viewport={"width": width, "height": height}, device_scale_factor=2)
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(HTML.as_uri(), wait_until="load")
            pg.wait_for_function("window.AVM_UI !== undefined", timeout=30000)
            for c in CASES:
                pg.select_option("#city", c["city"])
                pg.dispatch_event("#city", "change")
                pg.select_option("#district", c["district"])
                pg.dispatch_event("#district", "change")
                pg.select_option("#type", c["type"])
                pg.dispatch_event("#type", "change")
                pg.fill("#ask", str(c["ask"]))
                pg.dispatch_event("#ask", "input")
                js = pg.evaluate("""() => { const r = window.__lastResult;
                    return {pred: r.pred, fv: r.fv, lo: r.lo, hi: r.hi, gap: r.gap, p: r.p, light: r.light,
                            fvText: document.getElementById('o_fv').textContent,
                            rows: [...document.querySelectorAll('#o_comp tbody tr')].map(t => t.innerText)} }""")
                m = med.loc[c["type"]]
                inp = {"city": c["city"], "district": c["district"], "ward": "", "type": c["type"],
                       "area": float(m["area_m2"]), "bedrooms": int(m["bedrooms"]), "bathrooms": int(m["bathrooms"]),
                       "floors": int(m["floors"]), "frontage": float(m["frontage_m"]), "road": float(m["road_width_m"]),
                       "legal": "Sổ đỏ/Sổ hồng", "ask": c["ask"]}
                py = exp.app_value(inp, df, avm, conf, stale, lookup, scores)
                d_pred, d_p = abs(js["pred"] - py["pred"]), abs(js["p"] - py["p"])
                ok = d_pred < 1e-4 and d_p < 1e-4 and js["light"] == py["light"] and len(js["rows"]) == 5
                ok_all &= ok
                if tag == "desktop":
                    lines.append(f"{c['name']}: fair value {js['fv']:.2f} bn (Python {py['fv']:.2f}), range "
                                 f"{js['lo']:.2f} to {js['hi']:.2f} (Python {py['lo']:.2f} to {py['hi']:.2f}), gap "
                                 f"{js['gap']:+.1%} (Python {py['gap']:+.1%}), stale score {js['p']:.3f} "
                                 f"(Python {py['p']:.3f}), light {js['light']} (Python {py['light']}); "
                                 f"|d log price| {d_pred:.1e}, |d prob| {d_p:.1e}: {'OK' if ok else 'MISMATCH'}")
                pg.screenshot(path=str(FIG / f"dashboard_html_{tag}_{c['name']}.png"), full_page=True)
            # layout: no horizontal scrolling on the page
            overflow = pg.evaluate("() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
            lines.append(f"{tag} ({width} px): horizontal overflow {overflow} px")
            ok_all &= overflow <= 0
            pg.click("nav button[data-tab=mkt]")
            pg.wait_for_timeout(4000)
            pg.screenshot(path=str(FIG / f"dashboard_html_{tag}_market.png"), full_page=True)
            pg.click(".cityTog button[data-city='TP. Hồ Chí Minh']")
            pg.wait_for_timeout(3000)
            pg.screenshot(path=str(FIG / f"dashboard_html_{tag}_market_hcmc.png"), full_page=False)
            pg.click("nav button[data-tab=about]")
            pg.screenshot(path=str(FIG / f"dashboard_html_{tag}_about.png"), full_page=True)
            pg.close()
        b.close()
    lines.append(f"JavaScript errors: {len(errors)} {errors[:3]}")
    ok_all &= not errors
    lines.append(f"result: {'PASS' if ok_all else 'FAIL'}")
    (ROOT / "result" / "logs" / "dashboard_html_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    _ = np


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
