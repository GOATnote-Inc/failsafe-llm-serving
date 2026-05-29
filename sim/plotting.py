"""Minimal SVG charting — standard library only.

Why hand-rolled SVG instead of matplotlib? So the whole repo runs on a bare
Python interpreter (no pip install), and the charts render natively inside
GitHub markdown. Three chart types are enough for this project: a multi-line
time series, a stacked area, and grouped bars.
"""
from __future__ import annotations

import html
from typing import List, Optional, Sequence, Tuple

W, H = 940, 380
ML, MR, MT, MB = 70, 200, 44, 52  # margins (right margin holds the legend)
PLOT_W = W - ML - MR
PLOT_H = H - MT - MB

PALETTE = {
    "primary": "#1a9850", "fallback": "#66bd63", "retrieval": "#4575b4",
    "shed": "#fdae61", "failure": "#d73027", "offered": "#888888",
    "failsafe": "#1a9850", "naive": "#d73027", "slo": "#999999",
    "a": "#4575b4", "b": "#d73027", "c": "#fdae61", "kv": "#762a83",
}


def _nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    if hi <= lo:
        hi = lo + 1.0
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


class _Canvas:
    def __init__(self, title: str, xlabel: str, ylabel: str, xmax: float, ymax: float):
        self.s: List[str] = []
        self.xmax = max(1e-9, xmax)
        self.ymax = max(1e-9, ymax)
        self.s.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}" font-family="-apple-system,Segoe UI,Roboto,sans-serif">'
        )
        self.s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
        self.s.append(f'<text x="{ML}" y="24" font-size="16" font-weight="600">{html.escape(title)}</text>')
        # axes
        self.s.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PLOT_H}" stroke="#333"/>')
        self.s.append(f'<line x1="{ML}" y1="{MT+PLOT_H}" x2="{ML+PLOT_W}" y2="{MT+PLOT_H}" stroke="#333"/>')
        # y gridlines + labels
        for t in _nice_ticks(0, self.ymax):
            y = self.py(t)
            self.s.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+PLOT_W}" y2="{y:.1f}" stroke="#eee"/>')
            self.s.append(f'<text x="{ML-8}" y="{y+4:.1f}" font-size="11" text-anchor="end" fill="#555">{self._fmt(t)}</text>')
        # x ticks + labels
        for t in _nice_ticks(0, self.xmax):
            x = self.px(t)
            self.s.append(f'<text x="{x:.1f}" y="{MT+PLOT_H+18}" font-size="11" text-anchor="middle" fill="#555">{self._fmt(t)}</text>')
        self.s.append(f'<text x="{ML+PLOT_W/2}" y="{H-12}" font-size="12" text-anchor="middle" fill="#333">{html.escape(xlabel)}</text>')
        self.s.append(f'<text x="16" y="{MT+PLOT_H/2}" font-size="12" text-anchor="middle" fill="#333" transform="rotate(-90 16 {MT+PLOT_H/2})">{html.escape(ylabel)}</text>')
        self._legend_y = MT + 4

    @staticmethod
    def _fmt(v: float) -> str:
        if abs(v) >= 100:
            return f"{v:.0f}"
        if abs(v) >= 10:
            return f"{v:.0f}"
        if v == int(v):
            return f"{int(v)}"
        return f"{v:.1f}"

    def px(self, x: float) -> float:
        return ML + PLOT_W * (x / self.xmax)

    def py(self, y: float) -> float:
        return MT + PLOT_H * (1 - y / self.ymax)

    def legend(self, label: str, color: str, dashed: bool = False):
        x = ML + PLOT_W + 18
        y = self._legend_y
        dash = ' stroke-dasharray="5,4"' if dashed else ""
        self.s.append(f'<line x1="{x}" y1="{y}" x2="{x+22}" y2="{y}" stroke="{color}" stroke-width="3"{dash}/>')
        self.s.append(f'<text x="{x+28}" y="{y+4}" font-size="12" fill="#333">{html.escape(label)}</text>')
        self._legend_y += 20

    def shade(self, x0: float, x1: float, label: str):
        a, b = self.px(x0), self.px(x1)
        self.s.append(f'<rect x="{a:.1f}" y="{MT}" width="{b-a:.1f}" height="{PLOT_H}" fill="#000" opacity="0.05"/>')
        self.s.append(f'<text x="{(a+b)/2:.1f}" y="{MT+14}" font-size="11" text-anchor="middle" fill="#666">{html.escape(label)}</text>')

    def hline(self, y: float, label: str, color: str):
        yy = self.py(y)
        self.s.append(f'<line x1="{ML}" y1="{yy:.1f}" x2="{ML+PLOT_W}" y2="{yy:.1f}" stroke="{color}" stroke-width="1.5" stroke-dasharray="6,4"/>')
        self.s.append(f'<text x="{ML+PLOT_W-4}" y="{yy-5:.1f}" font-size="11" text-anchor="end" fill="{color}">{html.escape(label)}</text>')

    def polyline(self, xs: Sequence[float], ys: Sequence[float], color: str, width: float = 2.0):
        pts = " ".join(f"{self.px(x):.1f},{self.py(y):.1f}" for x, y in zip(xs, ys))
        self.s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"/>')

    def polygon(self, xs: Sequence[float], ys_top: Sequence[float], ys_bot: Sequence[float], color: str):
        top = " ".join(f"{self.px(x):.1f},{self.py(y):.1f}" for x, y in zip(xs, ys_top))
        bot = " ".join(f"{self.px(x):.1f},{self.py(y):.1f}" for x, y in zip(reversed(xs), reversed(ys_bot)))
        self.s.append(f'<polygon points="{top} {bot}" fill="{color}" opacity="0.85" stroke="none"/>')

    def bars(self, groups: List[str], series: List[Tuple[str, List[float], str]]):
        ng, ns = len(groups), len(series)
        gw = PLOT_W / max(1, ng)
        bw = gw * 0.7 / max(1, ns)
        for gi, g in enumerate(groups):
            gx = ML + gw * gi + gw * 0.15
            for si, (name, vals, color) in enumerate(series):
                v = vals[gi]
                x = gx + bw * si
                y = self.py(v)
                self.s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw-2:.1f}" height="{MT+PLOT_H-y:.1f}" fill="{color}"/>')
                self.s.append(f'<text x="{x+bw/2-1:.1f}" y="{y-4:.1f}" font-size="10" text-anchor="middle" fill="#333">{self._fmt(v)}</text>')
            self.s.append(f'<text x="{ML+gw*gi+gw/2:.1f}" y="{MT+PLOT_H+18}" font-size="11" text-anchor="middle" fill="#333">{html.escape(g)}</text>')

    def render(self) -> str:
        return "\n".join(self.s) + "\n</svg>\n"


def line_chart(path, title, x, series, xlabel, ylabel,
               hlines: Optional[List[Tuple[float, str, str]]] = None,
               shade: Optional[Tuple[float, float, str]] = None, ymax: Optional[float] = None):
    ym = ymax if ymax else max((max(s[1]) for s in series if s[1]), default=1.0) * 1.12
    c = _Canvas(title, xlabel, ylabel, max(x) if x else 1.0, ym)
    if shade:
        c.shade(*shade)
    for (y, label, color) in (hlines or []):
        c.hline(y, label, color)
    for (name, vals, color) in series:
        c.polyline(x, vals, color)
        c.legend(name, color)
    for (y, label, color) in (hlines or []):
        c.legend(label, color, dashed=True)
    _write(path, c.render())


def stacked_area(path, title, x, layers, xlabel, ylabel,
                 shade: Optional[Tuple[float, float, str]] = None, ymax: Optional[float] = None):
    n = len(x)
    cum = [0.0] * n
    totals = [sum(layer[1][i] for layer in layers) for i in range(n)]
    ym = ymax if ymax else (max(totals) if totals else 1.0) * 1.12
    c = _Canvas(title, xlabel, ylabel, max(x) if x else 1.0, ym)
    if shade:
        c.shade(*shade)
    for (name, vals, color) in layers:
        top = [cum[i] + vals[i] for i in range(n)]
        c.polygon(x, top, cum, color)
        cum = top
        c.legend(name, color)
    _write(path, c.render())


def grouped_bars(path, title, groups, series, ylabel, ymax: Optional[float] = None):
    ym = ymax if ymax else max((max(s[1]) for s in series if s[1]), default=1.0) * 1.18
    c = _Canvas(title, "", ylabel, len(groups), ym)
    c.bars(groups, series)
    for (name, _vals, color) in series:
        c.legend(name, color)
    _write(path, c.render())


def _write(path: str, content: str):
    with open(path, "w") as f:
        f.write(content)
