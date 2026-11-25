"""Draw the paper's vector PDF figures using Python 3 and ReportLab 4.

By default, run `python plots.py` beside the archived results. Supply
`--uci`, `--mnist`, and `--out` to use independently reproduced results.
"""

import argparse
import csv
import math
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas


WIDTH = 396
INK, MUTED, GRID = "#1D2832", "#435360", "#D5DBDF"
# Color and dash patterns encode the same method to support grayscale print.
METHODS = {
    "ISTA": ("#415165", (), "ISTA"),
    "FISTA": ("#B54A07", (3.8, 1.5), "FISTA"),
    "Restart-FISTA": ("#00638C", (5, 1.6, 0.8, 1.6), "Restart-FISTA"),
    "CD": ("#81519A", (0.85, 1.4), "CD"),
    "Restart+CD": ("#007057", (6.5, 1.5), "Restart+CD"),
}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def label(c, x, y, string, size=8, *, bold=False, align="left", color=INK):
    c.setFillColor(HexColor(color))
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    draw = {"left": c.drawString, "center": c.drawCentredString, "right": c.drawRightString}
    draw[align](x, y, string)


def pen(c, color, width, dash=()):
    c.setStrokeColor(HexColor(color))
    c.setLineWidth(width)
    c.setDash(dash)


def limit_and_step(value, intervals=4):
    """Choose human-readable ticks that cover all values of a positive axis."""
    raw_step = value / intervals
    unit = 10 ** math.floor(math.log10(raw_step))
    step = next(s * unit for s in (1, 2, 2.5, 5, 10) if s * unit >= raw_step)
    return math.ceil(value / step) * step, step


def convergence(curve_rows, destination):
    paths = {key: [] for key in METHODS}
    for row in curve_rows:
        if int(row["seed"]) == 11 and math.isclose(float(row["ratio"]), 0.1):
            paths[row["method"]].append(tuple(float(row[k]) for k in ("epoch", "elapsed_ms", "gap")))
    if any(not points for points in paths.values()):
        raise ValueError("Missing a method for the selected UCI problem")
    for points in paths.values():
        points.sort()

    c = canvas.Canvas(str(destination), pagesize=(WIDTH, 160), invariant=1)
    c.setTitle("Certified convergence on UCI digits, split 11")
    bottom, top = 33, 136
    gaps = [p[2] for points in paths.values() for p in points]
    if min(gaps) <= 0:
        raise ValueError("Logarithmic gap plot requires strictly positive gaps")
    log_lo = min(-8.6, math.log10(min(gaps)) - 0.2)
    log_hi = math.ceil(math.log10(max(gaps)))

    def gap_y(log_gap):
        return bottom + (top - bottom) * (log_gap - log_lo) / (log_hi - log_lo)

    panels = [
        (38, 201, 0, "Outer iteration (full sweep for CD)"),
        (223, 391, 1, "Elapsed solver time (ms)"),
    ]
    for left, right, column, axis_title in panels:
        max_value = max(p[column] for points in paths.values() for p in points)
        limit, step = limit_and_step(max_value, 5)

        def x_position(value):
            return left + (right - left) * value / limit

        for exponent in range(0, -9, -2):
            y = gap_y(exponent)
            if exponent == -8:
                pen(c, MUTED, 0.75, (2.5, 2.5))
            else:
                pen(c, GRID, 0.45)
            c.line(left, y, right, y)
            if column == 0:
                # Draw base and superscript separately for standard PDF fonts.
                tick_right = left - 3
                if exponent == 0:
                    label(c, tick_right, y - 2, "1", 7.4, align="right")
                else:
                    exp_text = str(exponent)
                    exp_width = c.stringWidth(exp_text, "Helvetica", 5.6)
                    label(c, tick_right - exp_width, y - 2, "10", 7.4, align="right")
                    label(c, tick_right, y + 1.4, exp_text, 5.6, align="right")

        pen(c, MUTED, 0.65)
        c.line(left, bottom, right, bottom)
        c.line(left, bottom, left, top)
        for tick in range(round(limit / step) + 1):
            value = tick * step
            x = x_position(value)
            c.line(x, bottom, x, bottom - 2.4)
            label(c, x, bottom - 11.7, f"{value:g}", 7.4, align="center")
        label(c, (left + right) / 2, 6, axis_title, 8.1, align="center")

        c.saveState()
        clipping = c.beginPath()
        clipping.rect(left, bottom, right - left, top - bottom)
        c.clipPath(clipping, stroke=0, fill=0)
        c.setLineJoin(1)
        for key, points in paths.items():
            color, dash, _ = METHODS[key]
            pen(c, color, 0.98, dash)
            line = c.beginPath()
            for i, point in enumerate(points):
                xy = x_position(point[column]), gap_y(math.log10(point[2]))
                (line.moveTo if i == 0 else line.lineTo)(*xy)
            c.drawPath(line)
            c.setFillColor(HexColor(color))
            c.circle(*xy, 1.45, stroke=0, fill=1)
        c.restoreState()

    c.saveState()
    c.translate(7, (bottom + top) / 2)
    c.rotate(90)
    label(c, 0, 0, "Primal–dual gap", align="center")
    c.restoreState()
    x = 26
    for color, dash, name in METHODS.values():
        pen(c, color, 1.3, dash)
        c.line(x, 151, x + 16, 151)
        label(c, x + 19, 148.5, name, 7.55)
        x += 19 + c.stringWidth(name, "Helvetica", 7.55) + 9
    pen(c, MUTED, 0.75, (2.5, 2.5))
    c.line(x, 151, x + 14, 151)
    label(c, x + 17, 148.5, "Tolerance", 7.55)
    c.save()


def paired_cases(uci_rows, mnist_rows):
    grouped = {}
    for dataset, rows in (("UCI", uci_rows), ("MNIST", mnist_rows)):
        for row in rows:
            key = dataset, row.get("seed", "Official"), float(row["ratio"])
            grouped.setdefault(key, {})[row["method"]] = row
    for (dataset, split, ratio), methods in sorted(
        grouped.items(), key=lambda p: (p[0][0] != "UCI", p[0][1], p[0][2])
    ):
        hybrid, restart = methods["Restart+CD"], methods["Restart-FISTA"]
        yield dataset, split, ratio, int(hybrid["restarts"]), (
            int(hybrid["epochs"]) / int(restart["epochs"]),
            float(hybrid["time_ms"]) / float(restart["time_ms"]),
        )


def metric_marker(c, x, y, *, time):
    fill = "#B54A07" if time else "#00638C"
    c.setFillColor(HexColor(fill))
    if time:
        diamond = c.beginPath()
        for i, (dx, dy) in enumerate(((0, 3), (3, 0), (0, -3), (-3, 0))):
            (diamond.moveTo if i == 0 else diamond.lineTo)(x + dx, y + dy)
        diamond.close()
        c.drawPath(diamond, stroke=0, fill=1)
    else:
        c.circle(x, y, 2.5, stroke=0, fill=1)


def tradeoff(cases, destination):
    c = canvas.Canvas(str(destination), pagesize=(WIDTH, 191), invariant=1)
    c.setTitle("Hybrid versus restarted FISTA for each measured problem")
    left, right = 107, 350
    low = math.floor(min(min(ratios) for *_, ratios in cases) * 10) / 10
    high = math.ceil(max(max(ratios) for *_, ratios in cases) * 10) / 10
    low, high = min(low, 0.9), max(high, 1.1)
    scale = lambda v: left + (right - left) * (v - low) / (high - low)

    metric_marker(c, 108, 184, time=False)
    label(c, 114, 181.5, "Outer iterations", 8)
    metric_marker(c, 185, 184, time=True)
    label(c, 191, 181.5, "Solver time", 8)
    headings = [(2, "Data", "left"), (37, "Split", "left"), (89, "Penalty", "right"),
                (394, "Sweeps", "right")]
    for x, name, align in headings:
        label(c, x, 166, name, 7.7, bold=True, align=align)
    label(c, scale(1), 166, "Parity", 7.7, bold=True, align="center")
    for i in range(round(10 * low), round(10 * high) + 1):
        v = i / 10
        pen(c, MUTED if i == 10 else GRID, 0.9 if i == 10 else 0.45)
        c.line(scale(v), 30.5, scale(v), 162)
        label(c, scale(v), 18, f"{v:.1f}", 7.3, align="center")
    previous = None
    for i, (dataset, split, ratio, sweeps, (iterations, elapsed)) in enumerate(cases):
        y = 155.5 - i * 11
        if previous != (dataset, split):
            if previous is not None:
                pen(c, GRID, 0.45)
                c.line(2, y + 5.5, 395, y + 5.5)
            if previous is None or previous[0] != dataset:
                label(c, 2, y - 2.5, dataset, 7.5, bold=True)
            label(c, 37, y - 2.5, split, 7.5)
        previous = (dataset, split)
        label(c, 89, y - 2.5, f"{ratio:.2f}", 7.5, align="right")
        label(c, 382, y - 2.5, str(sweeps), 7.5, align="right")
        pen(c, "#839099", 0.8)
        c.line(scale(iterations), y, scale(elapsed), y)
        metric_marker(c, scale(iterations), y, time=False)
        metric_marker(c, scale(elapsed), y, time=True)
    label(c, (left + right) / 2, 4, "Hybrid / restarted FISTA", 8.2, align="center")
    c.save()


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uci", type=Path, default=project / "results")
    parser.add_argument("--mnist", type=Path, default=project / "mnist")
    parser.add_argument("--out", type=Path, default=project / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    convergence(read_rows(args.uci / "curves.csv"), args.out / "convergence.pdf")
    cases = list(paired_cases(read_rows(args.uci / "summary.csv"), read_rows(args.mnist / "summary.csv")))
    if len(cases) != 12:
        raise ValueError(f"Expected 12 cases, found {len(cases)}")
    tradeoff(cases, args.out / "hybrid-tradeoff.pdf")


if __name__ == "__main__":
    main()
