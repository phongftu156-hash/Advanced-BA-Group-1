"""Phase 7: smoke-test the Streamlit app and take a screenshot for the report.

Starts `streamlit run code/app.py --server.headless true`, waits for it to answer, checks the
server log for errors, opens the page with Playwright (Chromium), enters an example asking
price and saves result/figures/dashboard_screenshot.png. If Playwright is unavailable, a
matplotlib mock-up labelled as such is saved instead.
"""
import subprocess
import sys
import shutil
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "result" / "figures"
LOG = ROOT / "result" / "logs" / "streamlit_smoke.txt"
PORT = 8599


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "code" / "app.py"),
                                 "--server.headless", "true", "--server.port", str(PORT),
                                 "--browser.gatherUsageStats", "false"], stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)
    ok = False
    try:
        for _ in range(60):
            time.sleep(1)
            try:
                if requests.get(f"http://localhost:{PORT}/_stcore/health", timeout=2).text.strip() == "ok":
                    ok = True
                    break
            except requests.RequestException:
                pass
        print("streamlit health:", "ok" if ok else "NOT RESPONDING")
        shot = False
        if ok:
            try:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as p:
                    b = p.chromium.launch(executable_path=shutil.which("chromium") or shutil.which("chromium-browser"))
                    # tall viewport so the whole page renders without inner scrolling; scale 2.67 for print
                    pg = b.new_page(viewport={"width": 1200, "height": 2000}, device_scale_factor=2.67)
                    pg.goto(f"http://localhost:{PORT}", wait_until="networkidle")
                    pg.wait_for_selector("text=AVM fair value", timeout=90000)
                    for label, text in (("District (as listed)", "Đống Đa"), ("Property type", "Alley house")):
                        pg.get_by_label(label).click()
                        time.sleep(0.5)
                        pg.keyboard.type(text)
                        time.sleep(0.5)
                        pg.keyboard.press("Enter")
                        time.sleep(2)
                    box = pg.get_by_label("Asking price (VND bn, optional; 0 = none)")
                    box.fill("9.5")
                    box.press("Enter")
                    pg.wait_for_selector("text=Asking price vs AVM value", timeout=60000)
                    time.sleep(6)
                    full = ROOT / "result" / "logs" / "dash_full.png"
                    pg.screenshot(path=str(full), full_page=True)
                    b.close()
                compose(full)
                shot = True
            except Exception as e:  # fall back to a labelled mock-up
                print("playwright screenshot failed:", e)
        time.sleep(max(0, 30 - 0))  # keep the server up for the 30-second smoke test
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    log = LOG.read_text(encoding="utf-8", errors="ignore")
    errors = [l for l in log.splitlines() if "Traceback" in l or "Error" in l]
    print("server log errors:", len(errors))
    for l in errors[:10]:
        print("  ", l)
    if not shot:
        mockup()
    return ok and not errors and shot


def compose(full, s=2.67):
    """Report exhibit: main results (left), segment map and the first inputs (right), browser margins cropped.
    Crop boxes are in CSS pixels of the 1200 px wide layout."""
    from PIL import Image, ImageDraw
    im = Image.open(full).convert("RGB")

    def crop(x0, y0, x1, y1):
        return im.crop((int(x0 * s), int(y0 * s), int(x1 * s), int(y1 * s)))
    main, mapc, side = crop(365, 105, 1130, 955), crop(375, 975, 1125, 1458), crop(12, 175, 288, 715)
    W, H = 765 + 24 + 480, 850
    canvas = Image.new("RGB", (int(W * s), int(H * s)), "white")
    canvas.paste(main, (0, 0))
    mw = int(480 * s)
    mh = int(mapc.height * mw / mapc.width)
    canvas.paste(mapc.resize((mw, mh), Image.LANCZOS), (int((765 + 24) * s), 0))
    sy = mh + int(18 * s)
    avail = int(H * s) - sy
    sc = min(avail / side.height, mw / side.width)
    side2 = side.resize((int(side.width * sc), int(side.height * sc)), Image.LANCZOS)
    bg = Image.new("RGB", (mw, avail), (240, 242, 246))
    bg.paste(side2, ((mw - side2.width) // 2, 0))
    canvas.paste(bg, (int((765 + 24) * s), sy))
    ImageDraw.Draw(canvas).rectangle([0, 0, canvas.width - 1, canvas.height - 1], outline=(210, 210, 210), width=4)
    canvas = canvas.resize((canvas.width * 3 // 4, canvas.height * 3 // 4), Image.LANCZOS)
    canvas.save(FIG / "dashboard_screenshot.png")


def mockup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, "MOCK-UP: dashboard screenshot unavailable", ha="center", va="center", fontsize=14)
    fig.savefig(FIG / "dashboard_screenshot.png", dpi=200)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
