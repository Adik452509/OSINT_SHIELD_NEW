#!/usr/bin/env python
"""M11 - build the report's figures and PDF.

    python scripts/12_build_report.py              # figures + PDF
    python scripts/12_build_report.py --figures    # figures only (needs runs/)
    python scripts/12_build_report.py --pdf        # PDF only

Figures (docs/figures/*.png) are drawn from the saved run results in runs/ -
gitignored, so they are committed as images. The PDF (docs/report.pdf) is
docs/REPORT.md rendered to HTML and printed by Microsoft Edge in headless mode,
which every Windows 11 machine has.

Needs:  pip install -e ".[report]"
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIGS = DOCS / "figures"
RUNS = ROOT / "runs"

CV_RUNS = [  # label, run folder under runs/m5 - the rows of REPORT.md §3.2
    ("mmBERT, keywords off", "mmBERT-small_body_only_kw-off_v2"),
    ("mmBERT, keywords on", "mmBERT-small_body_only_kw-group_counts_v2"),
    ("+ fixed 10 epochs", "mmBERT-small_body_only_kw-group_counts_m6-fixed"),
    ("+ 1,024 tokens", "mmBERT-small_body_only_kw-group_counts_m6-1024"),
    ("+ fixed 6 epochs (final)", "mmBERT-small_body_only_kw-group_counts_m6-e6"),
    ("XLM-R base, 10 epochs", "xlm-roberta-base_body_only_kw-group_counts_m6"),
    ("XLM-R base, 15 epochs", "xlm-roberta-base_body_only_kw-group_counts_m6-e15"),
]

INK, MUTED, ACCENT, TFIDF, RUBRIC, LIGHT = "#1b2a41", "#6b7a8c", "#4f46e5", "#e07b39", "#2e9e6b", "#c9d4e0"


def _style(plt):
    plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10, "axes.edgecolor": LIGHT,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.bbox": "tight"})


def figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    FIGS.mkdir(parents=True, exist_ok=True)

    # ---- fig 1: cross-validation, narrative 5-class -------------------------
    rows = []
    for label, folder in CV_RUNS:
        s = json.loads((RUNS / "m5" / folder / "summary.json").read_text())
        m = s["metrics"]["narrative_macro_f1"]
        rows.append((label, m["mean"], m["std"]))
    base = json.loads((RUNS / "baselines" / "baseline_results.json").read_text())
    tfidf = base["body_only|clean"]["tasks"]["narrative_5"]["tfidf_logreg"]["mean"]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = range(len(rows))[::-1]
    colors = [ACCENT if "final" in r[0] else LIGHT for r in rows]
    ax.barh(list(y), [r[1] for r in rows], xerr=[r[2] for r in rows], color=colors,
            error_kw={"ecolor": MUTED, "capsize": 3, "lw": 1}, height=0.6)
    ax.set_yticks(list(y), [r[0] for r in rows])
    ax.axvline(tfidf, color=TFIDF, ls="--", lw=1.4)
    ax.text(tfidf + 0.004, len(rows) - 0.45, f"TF-IDF {tfidf:.3f}", color=TFIDF, fontsize=9)
    for yi, (label, mean, _) in zip(y, rows):
        ax.text(0.012, yi, f"{mean:.3f}", va="center", fontsize=8.5,
                color="white" if "final" in label else INK)
    ax.set_xlim(0, 0.7)
    ax.set_xlabel("narrative 5-class macro-F1 (mean ± std, 15 runs)")
    ax.set_title("Cross-validation: every configuration, same folds and seeds",
                 loc="left", color=INK, fontsize=11)
    fig.savefig(FIGS / "fig1_cv.png")
    plt.close(fig)

    # ---- fig 2: held-out --------------------------------------------------------
    fr = json.loads((RUNS / "m7" / "final_results.json").read_text())
    enc_m, enc_s = fr["encoder"]["mean"]["bodied"], fr["encoder"]["std"]["bodied"]
    bl = fr["baselines"]
    cv = json.loads((RUNS / "m5" / CV_RUNS[4][1] / "summary.json").read_text())["metrics"]
    tasks = [("narrative 5-class", "narrative_macro_f1"),
             ("narrative 3-class", "narrative_collapsed_macro_f1"),
             ("severity F1(High)", "severity_f1_high")]
    series = [("encoder (3 seeds)", ACCENT, [enc_m[k] for _, k in tasks], [enc_s[k] for _, k in tasks]),
              ("TF-IDF", TFIDF, [bl["tfidf_logreg"]["bodied"][k] for _, k in tasks], None),
              ("keyword rubric", RUBRIC, [bl["keyword_rules"]["bodied"][k] for _, k in tasks], None),
              ("majority", LIGHT, [bl["majority"]["bodied"][k] for _, k in tasks], None)]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    width = 0.19
    for i, (name, color, vals, errs) in enumerate(series):
        xs = [t + (i - 1.5) * width for t in range(len(tasks))]
        ax.bar(xs, vals, width, color=color, label=name, yerr=errs,
               error_kw={"ecolor": INK, "capsize": 3, "lw": 1})
    ax.scatter(range(len(tasks)), [cv[k]["mean"] for _, k in tasks], marker="_", s=900, color=INK,
               zorder=5, label="encoder, CV estimate")
    ax.set_xticks(range(len(tasks)), [t for t, _ in tasks])
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("score")
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1),
              markerscale=0.35, handlelength=1.4, columnspacing=1.2)
    ax.set_title(f"Held-out, bodied articles (n = {fr['subset_n']['bodied']}) - scored once",
                 loc="left", color=INK, fontsize=11)
    fig.savefig(FIGS / "fig2_holdout.png")
    plt.close(fig)

    # ---- fig 3: accuracy by annotator agreement -----------------------------------
    ea = fr["error_analysis"]
    groups = ["annotators agreed", "annotators disagreed"]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for i, (task, key) in enumerate([("narrative", "narrative_acc_by_agreement"),
                                     ("severity", "severity_acc_by_agreement")]):
        vals = [ea[key][g]["accuracy"] for g in groups]
        ns = [ea[key][g]["n"] for g in groups]
        xs = [i - 0.18, i + 0.18]
        bars = ax.bar(xs, vals, 0.34, color=[ACCENT, TFIDF])
        for b, v, n in zip(bars, vals, ns):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1%}\n(n={n})", ha="center",
                    fontsize=8.5, color=INK)
    ax.set_xticks([0, 1], ["narrative accuracy", "severity accuracy"])
    ax.set_ylim(0, 1.12)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (ACCENT, TFIDF)],
              labels=groups, frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Deployed model on held-out articles: where the annotators agreed vs not",
                 loc="left", color=INK, fontsize=11)
    fig.savefig(FIGS / "fig3_agreement.png")
    plt.close(fig)
    print(f"  figures -> {FIGS.relative_to(ROOT)}")


CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: "Segoe UI", system-ui, sans-serif; font-size: 10pt; line-height: 1.45;
       color: #1b2a41; }
h1 { font-size: 20pt; margin: 0 0 4px; color: #1b2a41; }
h2 { font-size: 14pt; margin: 20px 0 6px; padding-bottom: 3px; border-bottom: 2px solid #4f46e5;
     page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14px 0 4px; color: #4f46e5; page-break-after: avoid; }
p, li { margin: 4px 0; }
a { color: #4f46e5; text-decoration: none; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 8.6pt; }
tr { page-break-inside: avoid; }
thead { display: table-header-group; }
th { background: #1b2a41; color: #fff; text-align: left; padding: 4px 6px; font-weight: 600; }
td { border-bottom: 1px solid #dde3ea; padding: 3px 6px; vertical-align: top; }
tr:nth-child(even) td { background: #f5f7fa; }
code { font-family: Consolas, monospace; font-size: 8.8pt; background: #eef1f5; padding: 0 3px;
       border-radius: 3px; }
pre { background: #f5f7fa; border: 1px solid #dde3ea; border-radius: 6px; padding: 8px 10px;
      font-size: 8pt; line-height: 1.35; page-break-inside: avoid; white-space: pre; overflow: hidden; }
pre code { background: none; padding: 0; font-size: 8pt; }
img { max-width: 100%; display: block; margin: 8px auto 12px; page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #dde3ea; margin: 14px 0; }
strong { color: #10192b; }
"""


def _edge() -> str | None:
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if Path(p).exists():
            return p
    return shutil.which("msedge") or shutil.which("chrome") or shutil.which("google-chrome")


def pdf() -> int:
    import markdown

    body = markdown.markdown((DOCS / "REPORT.md").read_text(encoding="utf-8"),
                             extensions=["tables", "fenced_code", "sane_lists"])
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<base href='{DOCS.as_uri()}/'><title>OSINT Shield - technical report</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    browser = _edge()
    if browser is None:
        print("  [!!] Microsoft Edge (or Chrome) not found - cannot print the PDF")
        return 1
    out = DOCS / "report.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={Path(tmp) / 'profile'}",
                        f"--print-to-pdf={out}", page.as_uri()],
                       check=False, timeout=180, capture_output=True)
    if not out.exists() or out.stat().st_size < 10_000:
        print("  [!!] the browser did not write the PDF")
        return 1
    print(f"  pdf     -> {out.relative_to(ROOT)}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--figures", action="store_true", help="figures only")
    ap.add_argument("--pdf", action="store_true", help="PDF only")
    args = ap.parse_args()
    both = not (args.figures or args.pdf)
    try:
        if args.figures or both:
            figures()
        if args.pdf or both:
            return pdf()
    except ImportError as exc:
        print(f"  [!!] {exc}\n       install the report extras:  pip install -e \".[report]\"")
        return 1
    except FileNotFoundError as exc:
        print(f"  [!!] {exc}\n       figures need the run results in runs/ (not in git)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
