"""Plot paired MNIST privacy controls from the archived CSVs (ReportLab 4)."""

import argparse
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas

from plots import GRID, MUTED, label, pen, read_rows


def paired_accuracies(directory):
    """Join the three fits by draw, converting proportions to percentages."""
    draws = {}
    baseline = None
    for row in read_rows(directory / "trials.csv"):
        budget = float(row["epsilon"])
        if budget == float("inf"):
            baseline = 100 * float(row["accuracy"])
        elif budget == 0.05:
            draws[int(row["draw"])] = {"joint": 100 * float(row["accuracy"])}
    for row in read_rows(directory / "component_trials.csv"):
        if float(row["epsilon"]) == 0.05:
            draws[int(row["draw"])][row["perturbed"]] = 100 * float(row["accuracy"])
    expected = {"joint", "h_only", "b_only"}
    if baseline is None or len(draws) != 30 or any(set(p) != expected for p in draws.values()):
        raise ValueError("Expected a clean baseline and 30 complete paired trials.")
    return baseline, draws


def symbol(c, x, y, component):
    if component == "b_only":  # Unperturbed h, noisy b.
        c.setFillColor(HexColor("#00638C"))
        c.circle(x, y, 2.15, stroke=0, fill=1)
    else:  # Noisy h, unperturbed b.
        pen(c, "#A84300", 0.95)
        c.setFillColor(white)
        path = c.beginPath()
        path.moveTo(x, y + 2.6)
        path.lineTo(x + 2.6, y)
        path.lineTo(x, y - 2.6)
        path.lineTo(x - 2.6, y)
        path.close()
        c.drawPath(path, stroke=1, fill=1)


def render(baseline, draws, destination):
    c = canvas.Canvas(str(destination), pagesize=(396, 190), invariant=1)
    c.setTitle("Paired controls for label perturbation at epsilon = 0.05")
    left, right, bottom, top = 33, 333, 30, 168
    # Limits cover the observed percentages with room for the baseline label.
    # Both axes show percentages; the equality line marks zero paired effect.
    xlow, xhigh, ylow, yhigh = 53, 78, 52, 85
    x = lambda v: left + (v - xlow) * (right - left) / (xhigh - xlow)
    y = lambda v: bottom + (v - ylow) * (top - bottom) / (yhigh - ylow)
    for tick in range(55, 86, 5):
        pen(c, GRID, 0.45)
        c.line(left, y(tick), right, y(tick))
        label(c, left - 4, y(tick) - 2.5, str(tick), 7.7, align="right")
    for tick in range(55, 76, 5):
        pen(c, MUTED, 0.6)
        c.line(x(tick), bottom, x(tick), bottom - 2.2)
        label(c, x(tick), bottom - 10.6, str(tick), 7.7, align="center")
    pen(c, MUTED, 0.6)
    c.line(left, bottom, left, top)
    c.line(left, bottom, right, bottom)
    pen(c, "#576672", 0.8, (4, 2.5))
    c.line(left, y(baseline), right, y(baseline))
    label(c, right + 4, y(baseline) - 2.4, f"Clean {baseline:.2f}%", 7.7)
    pen(c, MUTED, 0.68, (1, 1.7))
    c.line(left, y(xlow), right, y(xhigh))
    label(c, right + 4, y(xhigh) - 2.4, "No change", 7.7)

    for pair in draws.values():
        if not xlow <= pair["joint"] <= xhigh or any(
            not ylow <= pair[k] <= yhigh for k in ("b_only", "h_only")
        ):
            raise ValueError("An accuracy falls outside the figure axes.")
        pen(c, "#ADB9BF", 0.4)
        c.line(x(pair["joint"]), y(pair["h_only"]), x(pair["joint"]), y(pair["b_only"]))
    for pair in draws.values():
        for component in ("b_only", "h_only"):
            symbol(c, x(pair["joint"]), y(pair[component]), component)

    c.saveState()
    c.translate(8, (bottom + top) / 2)
    c.rotate(90)
    label(c, 0, 0, "Control test accuracy (%)", 8.3, align="center")
    c.restoreState()
    label(c, (left + right) / 2, 4, "Test accuracy with joint noise (%)", 8.3, align="center")

    for x_pos, component, name in [(61, "b_only", "Restore correlations (keep noisy intercept)"),
                                   (222, "h_only", "Restore intercept (keep noisy correlations)")]:
        symbol(c, x_pos, 181, component)
        label(c, x_pos + 6, 178.4, name, 7.65)
    c.save()


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--privacy", type=Path, default=root / "privacy")
    parser.add_argument("--out", type=Path, default=root / "figures" / "privacy-controls.pdf")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    render(*paired_accuracies(args.privacy), args.out)


if __name__ == "__main__":
    main()
