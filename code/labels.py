"""Collision-free label placement for scatter/bubble charts (display-space search)."""
import numpy as np


def place_labels(fig, ax, items, circles, avoid=(), fontsize=6.5, color="#232323", line_color="#6E6E6E"):
    """Place each label next to its point without touching bubbles, other labels or `avoid` artists.

    items   : list of (x, y, text, radius_px) in data coordinates; placed in list order (priority).
    circles : list of (x, y, radius_px) for every bubble on the chart (labelled or not).
    avoid   : artists (e.g. the legend) whose bounding boxes must stay free.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    inv = ax.transData.inverted()
    axbb = ax.get_window_extent(rend)
    blocked = [a.get_window_extent(rend) for a in avoid]
    circ = [(*to_px((x, y)), r) for x, y, r in circles]
    placed = []

    def hits(x0, y0, x1, y1):
        if x0 < axbb.x0 + 2 or x1 > axbb.x1 - 2 or y0 < axbb.y0 + 2 or y1 > axbb.y1 - 2:
            return True
        for b in blocked + placed:
            bx0, by0, bx1, by1 = (b.x0, b.y0, b.x1, b.y1) if hasattr(b, "x0") else b
            if x0 < bx1 and x1 > bx0 and y0 < by1 and y1 > by0:
                return True
        for cx, cy, r in circ:
            nx, ny = min(max(cx, x0), x1), min(max(cy, y0), y1)
            if (nx - cx) ** 2 + (ny - cy) ** 2 < (r + 1.5) ** 2:
                return True
        return False

    angles = np.deg2rad([0, 180, 30, 150, 330, 210, 60, 120, 300, 240, 90, 270])
    for x, y, text, r in items:
        t = ax.text(0, 0, text, fontsize=fontsize, color=color, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))
        bb = t.get_window_extent(rend)
        w, h = bb.width, bb.height
        px, py = to_px((x, y))
        best = None
        for d in [r + 3, r + 9, r + 16, r + 26, r + 38, r + 52, r + 70]:
            for a in angles:
                cx, cy = px + d * np.cos(a), py + d * np.sin(a)
                # anchor the box on the side facing the point
                x0 = cx if np.cos(a) > 0.3 else (cx - w if np.cos(a) < -0.3 else cx - w / 2)
                y0 = cy if np.sin(a) > 0.3 else (cy - h if np.sin(a) < -0.3 else cy - h / 2)
                if not hits(x0, y0, x0 + w, y0 + h):
                    best = (x0, y0, d, a)
                    break
            if best:
                break
        if best is None:  # fall back to the right of the point
            best = (px + r + 3, py - h / 2, r + 3, 0.0)
        x0, y0, d, a = best
        t.set_position(inv.transform((x0, y0)))
        t.set_va("bottom")
        t.set_ha("left")
        placed.append((x0, y0, x0 + w, y0 + h))
        if d > r + 4 or abs(np.sin(a)) > 0.1:  # leader line (offset or diagonal placement) from the bubble edge to the nearest point of the label box
            lx, ly = min(max(px, x0), x0 + w), min(max(py, y0), y0 + h)
            vx, vy = lx - px, ly - py
            n = max(np.hypot(vx, vy), 1e-9)
            sx, sy = px + vx / n * r, py + vy / n * r
            (dx0, dy0), (dx1, dy1) = inv.transform((sx, sy)), inv.transform((lx, ly))
            ax.plot([dx0, dx1], [dy0, dy1], color=line_color, lw=0.4, zorder=5)
