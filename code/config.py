"""Shared paths, constants and chart style for the pipeline."""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
EXTERNAL = DATA / "external"
OSM = EXTERNAL / "osm"
PROCESSED = DATA / "processed"
RESULT = ROOT / "result"
FIG = RESULT / "figures"
TAB = RESULT / "tables"
MODELS = RESULT / "models"
LOGS = RESULT / "logs"
for p in (OSM, PROCESSED, FIG, TAB, MODELS, LOGS):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
SCRAPE_DATE = "2026-10-15"
SOURCE_MOGI = "mogi.vn listings scraped Oct 2026"
SOURCE_OSM = "OpenStreetMap contributors (ODbL)"

# palette (matches the proposal)
BLUE = "#005AA0"
LBLUE = "#0078C8"
ORANGE = "#C85000"
DGREY = "#232323"
MGREY = "#6E6E6E"
LGREY = "#D2D2D2"
GREEN = "#288C46"
CLUSTER_COLORS = [BLUE, ORANGE, GREEN, "#7A4FA0", LBLUE, MGREY, "#B8860B", DGREY]

CITY_SHORT = {"Hà Nội": "Hanoi", "TP. Hồ Chí Minh": "HCMC"}
TYPE_EN = {
    "Căn hộ chung cư": "Apartment",
    "Nhà hẻm/ngõ": "Alley house",
    "Nhà mặt tiền/phố": "Street-front house",
    "Biệt thự/liền kề": "Villa / terraced",
    "Đất": "Land",
}
TYPE_ORDER = list(TYPE_EN)
CBD = {"Hà Nội": (21.0285, 105.8522), "TP. Hồ Chí Minh": (10.7725, 106.6980)}
# generous city bounding boxes (lat_min, lon_min, lat_max, lon_max); include merged Hà Tây / Củ Chi
BBOX = {"Hà Nội": (20.55, 105.25, 21.40, 106.05), "TP. Hồ Chí Minh": (10.35, 106.35, 11.20, 107.05)}

CM = 1 / 2.54
W1 = 8.6 * CM   # single column
W2 = 17.8 * CM  # full width


def get_logger(name):
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(LOGS / f"{name}.log", encoding="utf-8", mode="w")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(fh)
        log.addHandler(sh)
    return log


def set_style():
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MGREY,
        "axes.labelcolor": DGREY,
        "xtick.color": DGREY,
        "ytick.color": DGREY,
        "text.color": DGREY,
        "axes.grid": False,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    })


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    import matplotlib.pyplot as plt
    plt.close(fig)


def thousands(x, _=None):
    return f"{x:,.0f}"
