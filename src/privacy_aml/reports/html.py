"""Simple HTML validation report for synthetic data fidelity metrics."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must be before pyplot import
import matplotlib.pyplot as plt
import pandas as pd

_NUMERIC_COLS = ["amount_eur", "hour_of_day", "day_of_week"]
_CAT_COLS = [
    "currency", "channel", "country_code", "merchant_category", "is_suspicious"
]


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=80)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return data


def _histogram_html(col: str, real: pd.DataFrame, synth: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(real[col].dropna(), bins=50, alpha=0.5, label="real",
            color="steelblue", density=True)
    ax.hist(synth[col].dropna(), bins=50, alpha=0.5, label="synth",
            color="darkorange", density=True)
    ax.set_title(col)
    ax.legend()
    b64 = _fig_to_b64(fig)
    return (
        f'<div style="display:inline-block;margin:8px">'
        f'<img src="data:image/png;base64,{b64}"/></div>'
    )


def _barchart_html(col: str, real: pd.DataFrame, synth: pd.DataFrame) -> str:
    cats = sorted(real[col].astype(str).unique())
    r_freq = (
        real[col].astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
    )
    s_freq = (
        synth[col].astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
    )
    x = list(range(len(cats)))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(6, len(cats) * 0.5), 3))
    ax.bar(
        [i - width / 2 for i in x], r_freq.values,
        width=width, label="real", color="steelblue", alpha=0.8,
    )
    ax.bar(
        [i + width / 2 for i in x], s_freq.values,
        width=width, label="synth", color="darkorange", alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_title(col)
    ax.legend()
    b64 = _fig_to_b64(fig)
    return (
        f'<div style="display:inline-block;margin:8px">'
        f'<img src="data:image/png;base64,{b64}"/></div>'
    )


def _raw_table(results: dict) -> str:
    rows = []
    for col, v in results["ks_tests"].items():
        color = "green" if v["pass"] else "red"
        rows.append(
            f"<tr><td>KS</td><td>{col}</td>"
            f"<td>{v['D']:.4f}</td><td>{v['p']:.4f}</td>"
            f"<td style='color:{color}'>"
            f"<b>{'PASS' if v['pass'] else 'FAIL'}</b></td></tr>"
        )
    for col, v in results["chi2_tests"].items():
        color = "green" if v["pass"] else "red"
        rows.append(
            f"<tr><td>Chi²</td><td>{col}</td>"
            f"<td>{v['statistic']:.2f}</td><td>{v['p']:.4f}</td>"
            f"<td style='color:{color}'>"
            f"<b>{'PASS' if v['pass'] else 'FAIL'}</b></td></tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Test</th><th>Column</th><th>Statistic</th><th>p-value</th><th>Result</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def generate_report(
    results: dict,
    real: pd.DataFrame,
    synth: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a single-file HTML validation report with embedded charts.

    Args:
        results: Output of validate().
        real: Real transaction DataFrame (with hour_of_day, day_of_week columns).
        synth: Synthetic transaction DataFrame.
        output_path: Path to write the HTML file.
    """
    ratio = results["tstr_trts_ratio"]
    overall = results["overall_pass"]
    pass_color = "green" if overall else "red"
    pass_text = "PASS" if overall else "FAIL"
    ratio_color = "green" if ratio >= 0.85 else "red"

    ks_pass = sum(v["pass"] for v in results["ks_tests"].values())
    ks_total = len(results["ks_tests"])
    chi2_pass = sum(v["pass"] for v in results["chi2_tests"].values())
    chi2_total = len(results["chi2_tests"])

    summary = f"""
<div style="background:#f0f4f8;padding:16px;border-radius:8px;margin-bottom:24px">
  <h2 style="margin-top:0">Validation Summary</h2>
  <table style="border-collapse:collapse">
    <tr><td style="padding:4px 16px 4px 0">Overall</td>
        <td style="color:{pass_color}"><b style="font-size:1.2em">{pass_text}</b></td>
    </tr>
    <tr><td>TSTR AUC</td><td>{results['tstr_auc']:.4f}</td></tr>
    <tr><td>TRTS AUC</td><td>{results['trts_auc']:.4f}</td></tr>
    <tr><td>TSTR/TRTS Ratio</td>
        <td style="color:{ratio_color}"><b>{ratio:.4f}</b> (threshold ≥ 0.85)</td></tr>
    <tr><td>KS Tests Passed</td><td>{ks_pass}/{ks_total}</td></tr>
    <tr><td>Chi² Tests Passed</td><td>{chi2_pass}/{chi2_total}</td></tr>
  </table>
</div>"""

    numeric_section = "".join(_histogram_html(c, real, synth) for c in _NUMERIC_COLS)
    cat_section = "".join(_barchart_html(c, real, synth) for c in _CAT_COLS)
    raw_table = _raw_table(results)

    css = (
        "body{font-family:sans-serif;max-width:1200px;margin:auto;padding:24px}"
        "h1,h2{color:#1a2a3a}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:left}"
        "th{background:#eef2f5}"
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Synthetic Data Validation Report</title>
<style>{css}</style>
</head><body>
<h1>Synthetic Data Validation Report</h1>
{summary}
<h2>Numeric Distributions (KS Test — pass if D &lt; 0.1)</h2>
{numeric_section}
<h2>Categorical Distributions (Chi² Test — pass if p &gt; 0.05)</h2>
{cat_section}
<h2>Raw Test Results</h2>
{raw_table}
</body></html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"Report written → {output_path}")
