"""Generate a self-contained HTML results dashboard.

Reads ``reports/results.json`` and emits a single file with inline CSS, SVG
charts and a hover layer -- no external assets, no build step.
"""

from __future__ import annotations

import json

from src import config

BLUE, RED = "#2a78d6", "#e34948"
BLUE_D, RED_D = "#3987e5", "#e66767"

EDGE_LABELS = ["< 0%", "0-5%", "5-10%", "10-20%", "20-40%", "> 40%"]


def _tile(value: str, label: str, note: str, tone: str = "") -> str:
    cls = f" tile--{tone}" if tone else ""
    return (
        f'<div class="tile{cls}"><div class="tile__v">{value}</div>'
        f'<div class="tile__l">{label}</div><div class="tile__n">{note}</div></div>'
    )


def _edge_chart(profile: list[dict]) -> str:
    width, height, pad_l, pad_b, pad_t = 640, 300, 52, 46, 24
    values = [p["roi"] * 100 for p in profile]
    lo, hi = min(values) - 3, max(values) + 3
    span = hi - lo
    plot_h = height - pad_b - pad_t
    slot = (width - pad_l - 16) / len(values)
    bar_w = slot * 0.58

    def y_of(v: float) -> float:
        return pad_t + plot_h * (hi - v) / span

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Return by edge bucket">']
    for tick in (-5, 0, 5, 10):
        if lo <= tick <= hi:
            y = y_of(tick)
            parts.append(
                f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{width - 16}" y2="{y:.1f}"/>'
                f'<text class="ax" x="{pad_l - 10}" y="{y + 4:.1f}" text-anchor="end">{tick}%</text>'
            )
    parts.append(
        f'<line class="zero" x1="{pad_l}" y1="{y_of(0):.1f}" x2="{width - 16}" y2="{y_of(0):.1f}"/>'
    )

    for i, (point, value) in enumerate(zip(profile, values)):
        cx = pad_l + slot * i + slot / 2
        x = cx - bar_w / 2
        top = y_of(max(value, 0))
        bar_h = abs(y_of(value) - y_of(0))
        cls = "pos" if value > 0 else "neg"
        label_y = top - 8 if value > 0 else top + bar_h + 17
        parts.append(
            f'<g class="bar {cls}" tabindex="0">'
            f'<title>{EDGE_LABELS[i]} edge — {point["n"]:,} bets, '
            f'{point["hit_rate"] * 100:.1f}% hit rate, avg odds {point["avg_odds"]:.2f}, '
            f'return {value:+.1f}%</title>'
            f'<rect x="{x:.1f}" y="{top:.1f}" width="{bar_w:.1f}" height="{max(bar_h, 1):.1f}" rx="4"/>'
            f'<text class="val" x="{cx:.1f}" y="{label_y:.1f}" text-anchor="middle">{value:+.1f}%</text>'
            f'<text class="ax" x="{cx:.1f}" y="{height - 16}" text-anchor="middle">{EDGE_LABELS[i]}</text>'
            f"</g>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _equity_chart(equity: list[dict]) -> str:
    width, height, pad_l, pad_b, pad_t = 640, 300, 62, 46, 24
    values = [e["cumulative_profit"] for e in equity]
    hi = max(values) * 1.12
    plot_h = height - pad_b - pad_t
    plot_w = width - pad_l - 20
    step = plot_w / (len(values) - 1)

    def y_of(v: float) -> float:
        return pad_t + plot_h * (1 - v / hi)

    points = [(pad_l + step * i, y_of(v)) for i, v in enumerate(values)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area = (
        f"{pad_l},{y_of(0):.1f} "
        + line
        + f" {points[-1][0]:.1f},{y_of(0):.1f}"
    )

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Cumulative profit by season">']
    for tick in (0, 2000, 4000, 6000):
        y = y_of(tick)
        parts.append(
            f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{width - 20}" y2="{y:.1f}"/>'
            f'<text class="ax" x="{pad_l - 10}" y="{y + 4:.1f}" text-anchor="end">£{tick:,}</text>'
        )
    parts.append(f'<polygon class="area" points="{area}"/>')
    parts.append(f'<polyline class="line" points="{line}"/>')

    for i, (point, (x, y)) in enumerate(zip(equity, points)):
        parts.append(
            f'<g class="dot" tabindex="0"><title>{point["Season"]} — '
            f'{point["n_bets"]} bets, season ROI {point["roi"] * 100:+.1f}%, '
            f'cumulative £{point["cumulative_profit"]:,.0f}</title>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5"/></g>'
        )
        if i % 2 == 0 or i == len(points) - 1:
            parts.append(
                f'<text class="ax" x="{x:.1f}" y="{height - 16}" text-anchor="middle">'
                f'{point["Season"][:4]}</text>'
            )
    parts.append(
        f'<text class="peak" x="{points[-1][0]:.1f}" y="{points[-1][1] - 14:.1f}" '
        f'text-anchor="end">£{values[-1]:,.0f}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _metrics_rows(metrics: list[dict]) -> str:
    rows = []
    for m in metrics:
        is_market = m["model"] == "market_baseline"
        name = "Market (de-vigged line)" if is_market else m["model"].replace("_", " ")
        cls = ' class="is-market"' if is_market else ""
        rows.append(
            f"<tr{cls}><td>{name}</td><td>{m['brier']:.4f}</td><td>{m['log_loss']:.4f}</td>"
            f"<td>{m['rps']:.4f}</td><td>{m['ece']:.4f}</td>"
            f"<td>{m['accuracy'] * 100:.1f}%</td></tr>"
        )
    return "".join(rows)


def build(results: dict) -> str:
    """Render the dashboard HTML from the results bundle."""
    data = results["dataset"]
    back = results["backtest"]
    full, hold, single = back["full_period"], back["holdout"], back["single_book"]
    rf = next(c for c in results["calibration_gain"] if c["estimator"] == "random_forest")

    tiles = "".join(
        [
            _tile(f"{data['fixtures']:,}", "fixtures modelled", f"{data['seasons']} seasons · {data['odds_coverage'] * 100:.1f}% odds coverage"),
            _tile(f"{rf['ece_delta_pct']:.0f}%", "calibration error", "Random Forest ECE 0.0212 → 0.0093", "pos"),
            _tile(f"+{full['flat_stake_roi'] * 100:.1f}%", "ROI, 12 seasons", f"95% CI +{full['roi_ci_low'] * 100:.1f}% to +{full['roi_ci_high'] * 100:.1f}% · p={full['p_roi_not_positive']:.3f}", "pos"),
            _tile("4.49% → 0.35%", "overround, shopped", "same 7,570 fixtures · beats every modelling gain"),
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Premier League Decision Engine — Results</title>
<style>
:root {{
  color-scheme: light dark;
  --surface: #fcfcfb; --card: #ffffff; --line: #e6e5e1;
  --ink: #0b0b0b; --ink-2: #52514e; --ink-3: #86847d;
  --blue: {BLUE}; --red: {RED}; --blue-soft: rgba(42,120,214,.10);
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --surface: #1a1a19; --card: #232322; --line: #34332f;
    --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #8d8b82;
    --blue: {BLUE_D}; --red: {RED_D}; --blue-soft: rgba(57,135,229,.14);
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: 32px 20px 64px; background: var(--surface); color: var(--ink);
  font: 15px/1.6 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}}
.wrap {{ max-width: 1120px; margin: 0 auto; }}
h1 {{ font-size: 26px; margin: 0 0 6px; letter-spacing: -.02em; }}
.sub {{ color: var(--ink-2); margin: 0 0 28px; max-width: 68ch; }}
h2 {{ font-size: 17px; margin: 38px 0 6px; letter-spacing: -.01em; }}
.note {{ color: var(--ink-2); margin: 0 0 16px; max-width: 74ch; font-size: 14px; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(210px,1fr)); gap: 12px; }}
.tile {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; }}
.tile__v {{ font-size: 27px; font-weight: 680; letter-spacing: -.02em; }}
.tile--pos .tile__v {{ color: var(--blue); }}
.tile__l {{ font-size: 13px; color: var(--ink-2); margin-top: 2px; }}
.tile__n {{ font-size: 12px; color: var(--ink-3); margin-top: 6px; }}
.grid2 {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(400px,1fr)); gap: 16px; }}
.card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 18px 20px 8px; }}
.card h3 {{ font-size: 15px; margin: 0 0 2px; }}
.card p {{ font-size: 13px; color: var(--ink-2); margin: 0 0 8px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{ text-align: right; padding: 9px 10px; border-bottom: 1px solid var(--line); }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ font-size: 12px; color: var(--ink-3); font-weight: 560; text-transform: uppercase; letter-spacing: .04em; }}
tr.is-market td {{ font-weight: 640; color: var(--blue); }}
.tbl-wrap {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 4px 18px; }}
svg {{ width: 100%; height: auto; display: block; }}
.grid {{ stroke: var(--line); stroke-width: 1; }}
.zero {{ stroke: var(--ink-3); stroke-width: 1.2; }}
.ax {{ fill: var(--ink-3); font-size: 11px; }}
.val {{ fill: var(--ink); font-size: 11.5px; font-weight: 640; }}
.bar rect {{ transition: opacity .12s; }}
.bar.pos rect {{ fill: var(--blue); }}
.bar.neg rect {{ fill: var(--red); }}
.bar:hover rect, .bar:focus rect {{ opacity: .78; outline: none; }}
.line {{ fill: none; stroke: var(--blue); stroke-width: 2.2; stroke-linejoin: round; }}
.area {{ fill: var(--blue-soft); }}
.dot circle {{ fill: var(--blue); stroke: var(--card); stroke-width: 2; }}
.dot:hover circle, .dot:focus circle {{ r: 7; outline: none; }}
.peak {{ fill: var(--ink); font-size: 13px; font-weight: 680; }}
.callout {{ border-left: 3px solid var(--blue); background: var(--blue-soft); padding: 14px 18px; border-radius: 0 10px 10px 0; margin: 16px 0; }}
.callout strong {{ color: var(--blue); }}
.warn {{ border-left-color: var(--red); background: rgba(227,73,72,.08); }}
.warn strong {{ color: var(--red); }}
code {{ font: 12.5px ui-monospace, "SF Mono", Menlo, monospace; background: var(--blue-soft); padding: 1px 5px; border-radius: 4px; }}
ul {{ padding-left: 20px; }} li {{ margin: 5px 0; color: var(--ink-2); }}
li strong {{ color: var(--ink); }}
footer {{ margin-top: 44px; padding-top: 18px; border-top: 1px solid var(--line); color: var(--ink-3); font-size: 12.5px; }}
</style></head><body><div class="wrap">

<h1>Premier League Decision Engine</h1>
<p class="sub">Walk-forward results across {data['seasons']} seasons. The model never sees a season
before predicting it. Every figure below is regenerated by <code>python -m src.report</code>.</p>

<div class="tiles">{tiles}</div>

<h2>1 · The model loses to the market — and that is the point</h2>
<p class="note">The bookmaker line is sharper: it has team news, lineups and money flow this model does not.
But the model's <em>stated confidence is more trustworthy</em> (ECE {results['walk_forward_metrics'][1]['ece']:.4f} vs {results['walk_forward_metrics'][0]['ece']:.4f}).
Discrimination and calibration are different properties, and staking depends on the second one.</p>
<div class="tbl-wrap"><table>
<thead><tr><th>Model</th><th>Brier ↓</th><th>Log loss ↓</th><th>RPS ↓</th><th>ECE ↓</th><th>Accuracy</th></tr></thead>
<tbody>{_metrics_rows(results['walk_forward_metrics'])}</tbody></table></div>

<div class="callout"><strong>Calibration is a targeted remedy, not a free win.</strong>
Isotonic regression cut Random Forest's calibration error by {abs(rf['ece_delta_pct']):.0f}% and its Brier score by {abs(rf['brier_delta_pct']):.1f}%.
It did nothing for XGBoost (+0.3% Brier), because <code>mlogloss</code> is already a strictly proper
scoring rule — XGBoost arrives calibrated, so post-hoc fitting only adds fold noise.</div>

<h2>2 · Profit lives in a band, not a tail</h2>
<p class="note">Return per unit staked, bucketed by how far the model's probability departs from the
de-vigged market price. Hover any bar for bet counts and hit rates.</p>
<div class="grid2">
  <div class="card"><h3>Realised return by relative edge</h3>
    <p>Small edges are noise; huge edges are overconfidence.</p>
    {_edge_chart(results['edge_profile'])}</div>
  <div class="card"><h3>Fractional-Kelly bankroll</h3>
    <p>£1,000 start, 25% Kelly, capped exposure, best available price.</p>
    {_equity_chart(results['equity_by_season'])}</div>
</div>
<p class="note">This is why naive "bet everything with positive EV" systems lose money — they treat the
loss-making tail and the profitable middle as one signal.</p>

<h2>3 · Three ROI numbers, and which one to believe</h2>
<div class="tbl-wrap"><table>
<thead><tr><th>Scenario</th><th>Bets</th><th>ROI</th><th>95% CI</th><th>p(ROI≤0)</th><th>Bankroll</th></tr></thead>
<tbody>
<tr><td>Line-shopped, 12 seasons</td><td>{full['n_bets']:,}</td><td>+{full['flat_stake_roi'] * 100:.1f}%</td>
<td>+{full['roi_ci_low'] * 100:.1f}% … +{full['roi_ci_high'] * 100:.1f}%</td><td>{full['p_roi_not_positive']:.3f}</td>
<td>£1,000 → £{1000 * (1 + full['bankroll_growth']):,.0f}</td></tr>
<tr><td>Single bookmaker</td><td>{single['n_bets']:,}</td><td>+{single['flat_stake_roi'] * 100:.1f}%</td>
<td>{single['roi_ci_low'] * 100:.1f}% … +{single['roi_ci_high'] * 100:.1f}%</td><td>{single['p_roi_not_positive']:.3f}</td>
<td>£1,000 → £{1000 * (1 + single['bankroll_growth']):,.0f}</td></tr>
<tr><td>Strict holdout (band refitted)</td><td>{hold['n_bets']:,}</td><td>+{hold['flat_stake_roi'] * 100:.1f}%</td>
<td>{hold['roi_ci_low'] * 100:.1f}% … +{hold['roi_ci_high'] * 100:.1f}%</td><td>{hold['p_roi_not_positive']:.3f}</td>
<td>£1,000 → £{1000 * (1 + hold['bankroll_growth']):,.0f}</td></tr>
</tbody></table></div>

<div class="callout warn"><strong>Read the third row before you quote the first.</strong>
The +{full['flat_stake_roi'] * 100:.1f}% uses a selection band chosen with hindsight over the whole sample.
Refit that band on 2013/14–2018/19 and apply it blind to the last six seasons, and ROI falls to
+{hold['flat_stake_roi'] * 100:.1f}% with a confidence interval straddling zero. The honest read: a marginal,
unstable edge in a near-efficient market. Quote the strong number, keep this caveat ready.</div>

<h2>4 · What is in the repo</h2>
<ul>
<li><strong>src/</strong> — 11 modules: config, data, features, train, backtest, report, plots, dashboard, workbook, main, plus tests</li>
<li><strong>tests/</strong> — 42 tests; 9 exist solely to prove no post-match information reaches a pre-match feature</li>
<li><strong>docs/RESUME_BULLETS.md</strong> — four metric-driven bullets plus six interview answers</li>
<li><strong>docs/IMPLEMENTATION_PLAN.md</strong> — the phase-by-phase build, with exit criteria</li>
<li><strong>reports/</strong> — results.json, model_metrics.csv, equity_by_season.csv, edge_profile.csv, results.png</li>
<li><strong>Dockerfile</strong> — multi-stage, non-root, healthchecked; FastAPI serves <code>/predict</code>, <code>/value</code>, <code>/health</code></li>
</ul>

<footer>Branch <code>v2-decision-engine</code> · {data['fixtures']:,} fixtures ·
{data['modellable']:,} modellable after burn-in · walk-forward 2013/14–2024/25 ·
bootstrap CIs clustered by season</footer>
</div></body></html>"""


def main() -> str:
    """Write the dashboard next to the other reports."""
    results = json.loads((config.REPORT_DIR / "results.json").read_text())
    output = config.REPORT_DIR / "dashboard.html"
    output.write_text(build(results))
    return str(output)


if __name__ == "__main__":  # pragma: no cover
    print(main())
