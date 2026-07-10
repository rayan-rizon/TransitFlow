#!/usr/bin/env python3
"""Generate MNRAS figures with ReportLab (no GUI/scientific binary backend)."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.graphics import renderPDF
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, Group, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BLUE = HexColor("#0072B2")
ORANGE = HexColor("#E69F00")
GREEN = HexColor("#009E73")


def read_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def write_drawing(drawing: Drawing, stem: str) -> None:
    renderPDF.drawToFile(drawing, str(OUT / f"{stem}.pdf"))


def add_text(d: Drawing, x: float, y: float, text: str, size: float = 7,
             anchor: str = "middle", bold: bool = False,
             angle: float = 0) -> None:
    font = "Helvetica-Bold" if bold else "Helvetica"
    if angle:
        group = Group()
        group.add(String(0, 0, text, fontName=font, fontSize=size,
                         textAnchor=anchor, fillColor=colors.black))
        group.transform = (0, 1, -1, 0, x, y)
        d.add(group)
    else:
        d.add(String(x, y, text, fontName=font, fontSize=size,
                     textAnchor=anchor, fillColor=colors.black))


def calibration_figure() -> None:
    metrics = read_json(
        ROOT / "artifacts/canonical/vast_publishable_bf16_5090_20260708_v3/results/synthetic/metrics.json"
    )
    levels = [float(x) for x in metrics["coverage_levels"]]
    coverage = [float(x) for x in metrics["characterization_coverage_overall"]]
    impact = metrics["stratified_characterization"]["impact"]

    d = Drawing(7.15 * inch, 3.05 * inch)
    plot = LinePlot()
    plot.x, plot.y, plot.width, plot.height = 45, 38, 195, 150
    plot.data = [list(zip([0.0, 1.0], [0.0, 1.0])), list(zip(levels, coverage))]
    plot.xValueAxis.valueMin, plot.xValueAxis.valueMax = 0, 1
    plot.yValueAxis.valueMin, plot.yValueAxis.valueMax = 0, 1
    plot.xValueAxis.valueSteps = [0, .2, .4, .6, .8, 1]
    plot.yValueAxis.valueSteps = [0, .2, .4, .6, .8, 1]
    plot.lines[0].strokeColor = colors.grey
    plot.lines[0].strokeDashArray = [4, 3]
    plot.lines[0].strokeWidth = 0.8
    plot.lines[1].strokeColor = BLUE
    plot.lines[1].strokeWidth = 1.4
    plot.lines[1].symbol = None
    d.add(plot)
    add_text(d, 142, 12, "Nominal credible level")
    add_text(d, 12, 113, "Empirical coverage", angle=90)
    add_text(d, 48, 194, "(a)", bold=True)
    add_text(d, 143, 202, "Pooled expected coverage", size=8, bold=True)

    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = 300, 38, 195, 150
    keys = ["low_b", "mid_b", "high_b"]
    chart.data = [
        [float(impact[k]["RpRs_cov68"]) for k in keys],
        [float(impact[k]["aRs_cov68"]) for k in keys],
        [float(impact[k]["b_cov68"]) for k in keys],
    ]
    chart.categoryAxis.categoryNames = ["low b", "mid b", "high b"]
    chart.valueAxis.valueMin, chart.valueAxis.valueMax = 0.4, 0.82
    chart.valueAxis.valueStep = 0.1
    chart.bars[0].fillColor = BLUE
    chart.bars[1].fillColor = ORANGE
    chart.bars[2].fillColor = GREEN
    chart.barSpacing = 1
    chart.groupSpacing = 5
    chart.categoryAxis.labels.fontSize = 6.5
    chart.valueAxis.labels.fontSize = 6.5
    d.add(chart)
    y68 = chart.y + (0.68 - chart.valueAxis.valueMin) / (
        chart.valueAxis.valueMax - chart.valueAxis.valueMin) * chart.height
    target = Line(chart.x, y68, chart.x + chart.width, y68,
                  strokeColor=colors.darkgrey, strokeWidth=0.8)
    target.strokeDashArray = [4, 3]
    d.add(target)
    add_text(d, 398, 12, "Impact-parameter stratum")
    add_text(d, 272, 113, "Empirical 68% coverage", angle=90)
    add_text(d, 303, 194, "(b)", bold=True)
    add_text(d, 398, 202, "Conditional coverage", size=8, bold=True)
    for x, colour, label in ((315, BLUE, "Rp/Rs"), (372, ORANGE, "a/Rs"),
                             (429, GREEN, "b")):
        d.add(Rect(x, 184, 8, 5, fillColor=colour, strokeColor=None))
        add_text(d, x + 12, 184, label, size=6.2, anchor="start")
    write_drawing(d, "calibration_diagnostics")


def detection_figure() -> None:
    preferred = ROOT / "artifacts/fair_candidate_validation_n1000.json"
    report = read_json(preferred if preferred.exists() else
                       ROOT / "artifacts/fair_candidate_pilot_n200.json")
    ci = report["uncertainty"]["ci95"]
    d = Drawing(3.5 * inch, 3.05 * inch)
    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = 42, 43, 188, 145
    chart.data = [
        [float(report["bls"]["roc_auc"]), float(report["transitflow"]["roc_auc"])],
        [float(report["bls"]["average_precision"]),
         float(report["transitflow"]["average_precision"])],
    ]
    chart.categoryAxis.categoryNames = ["BLS", "TransitFlow"]
    chart.valueAxis.valueMin, chart.valueAxis.valueMax = 0.45, 1.0
    chart.valueAxis.valueStep = 0.1
    chart.bars[0].fillColor = BLUE
    chart.bars[1].fillColor = ORANGE
    chart.barSpacing = 2
    chart.groupSpacing = 16
    chart.categoryAxis.labels.fontSize = 7
    chart.valueAxis.labels.fontSize = 6.5
    d.add(chart)
    add_text(d, 136, 13, "Candidate-scoring method")
    add_text(d, 14, 114, "Discrimination metric", angle=90)
    add_text(d, 136, 203, "Fair BLS-candidate benchmark", size=8, bold=True)
    for x, colour, label in ((62, BLUE, "ROC-AUC"), (135, ORANGE, "Average precision")):
        d.add(Rect(x, 190, 8, 5, fillColor=colour, strokeColor=None))
        add_text(d, x + 12, 190, label, size=6.2, anchor="start")
    add_text(d, 230, 194, f"n={report['n']}", size=6.5, anchor="end")

    # Draw paired-bootstrap 95% intervals over the centres of the grouped bars.
    centres = {
        "bls_auc": 42 + 188 * 0.25 - 9,
        "bls_ap": 42 + 188 * 0.25 + 9,
        "tf_auc": 42 + 188 * 0.75 - 9,
        "tf_ap": 42 + 188 * 0.75 + 9,
    }
    for key, x in centres.items():
        lo, hi = ci[key]
        y0 = chart.y + (lo - 0.45) / 0.55 * chart.height
        y1 = chart.y + (hi - 0.45) / 0.55 * chart.height
        d.add(Line(x, y0, x, y1, strokeColor=colors.black, strokeWidth=0.8))
        d.add(Line(x - 3, y0, x + 3, y0, strokeColor=colors.black, strokeWidth=0.8))
        d.add(Line(x - 3, y1, x + 3, y1, strokeColor=colors.black, strokeWidth=0.8))
    write_drawing(d, "candidate_detection")


def architecture_figure() -> None:
    d = Drawing(7.15 * inch, 2.15 * inch)
    boxes = [
        (10, 55, 78, 48, "BLS candidate", "P, t0"),
        (112, 35, 92, 88, "Three views", "global / local / periodogram"),
        (235, 50, 82, 58, "Shared CNN", "tri-branch embedding"),
        (350, 88, 90, 38, "Detection head", "p(d=1 | x,c)"),
        (350, 30, 90, 38, "FMPE head", "p(theta | d=1,x,c)"),
        (470, 55, 42, 48, "Score +", "posterior"),
    ]
    fills = [HexColor("#DDEBF7"), HexColor("#E2F0D9"), HexColor("#FFF2CC"),
             HexColor("#FCE4D6"), HexColor("#E4DFEC"), HexColor("#D9EAD3")]
    for (x, y, w, h, line1, line2), fill in zip(boxes, fills):
        d.add(Rect(x, y, w, h, fillColor=fill, strokeColor=colors.darkgrey,
                   strokeWidth=0.8))
        add_text(d, x + w / 2, y + h / 2 + 5, line1, size=7, bold=True)
        add_text(d, x + w / 2, y + h / 2 - 7, line2, size=6.2)

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        d.add(Line(x0, y0, x1, y1, strokeColor=colors.darkgrey, strokeWidth=1))
        d.add(Polygon([x1, y1, x1 - 5, y1 + 2.5, x1 - 5, y1 - 2.5],
                      fillColor=colors.darkgrey, strokeColor=None))

    arrow(88, 79, 112, 79)
    arrow(204, 79, 235, 79)
    arrow(317, 79, 350, 107)
    arrow(317, 79, 350, 49)
    arrow(440, 107, 470, 87)
    arrow(440, 49, 470, 70)
    write_drawing(d, "architecture")


if __name__ == "__main__":
    calibration_figure()
    detection_figure()
    architecture_figure()
