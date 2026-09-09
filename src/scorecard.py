"""Public scorecard: what the model predicted, and what actually happened.

Every prediction in this report was made on a match the model had never seen.
The source is ``data/processed/oos_predictions.parquet``, written by the
walk-forward loop in :mod:`src.train`: for season N the model is trained only on
seasons before N, so nothing here is fitted on its own answer.

Run with ``python -m src.scorecard``. Output: ``reports/scorecard.html``.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

VARIANT = "xgb_uncalibrated"  # the model actually shipped
OUTPUT = config.REPORT_DIR / "scorecard.html"

# Categorical pair, validated for CVD separation against both surfaces.
INK_LIGHT, INK_DARK = "#0B6E9B", "#3795BC"  # the model
MKT_LIGHT, MKT_DARK = "#C2571A", "#C87A3E"  # the bookmakers


# --------------------------------------------------------------------------- #
# Computation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Scorecard:
    n: int
    seasons: int
    span: str
    hit: float
    market_hit: float
    brier: float
    market_brier: float
    closer: float
    reliability: pd.DataFrame
    by_season: pd.DataFrame
    misses: pd.DataFrame
    best: pd.DataFrame


def load() -> pd.DataFrame:
    preds = pd.read_parquet(config.PROCESSED_DIR / "oos_predictions.parquet")
    store = pd.read_parquet(config.FEATURE_STORE_FILE)
    keep = [
        "match_id", "mkt_prob_home", "mkt_prob_draw", "mkt_prob_away",
        "FullTimeHomeGoals", "FullTimeAwayGoals",
    ]
    return preds.loc[preds.variant == VARIANT].merge(store[keep], on="match_id")


def compute(frame: pd.DataFrame) -> Scorecard:
    model = frame[["prob_home", "prob_draw", "prob_away"]].to_numpy()
    market = frame[["mkt_prob_home", "mkt_prob_draw", "mkt_prob_away"]].to_numpy()
    truth = frame["target"].to_numpy()
    onehot = np.eye(3)[truth]

    def brier(probabilities: np.ndarray) -> float:
        return float(((probabilities - onehot) ** 2).sum(1).mean())

    picked, market_picked = model.argmax(1), market.argmax(1)

    # Reliability: pool every probability the model ever quoted, bucket it, and
    # ask how often the thing it was quoted about actually happened.
    quoted = pd.DataFrame({"said": model.ravel(), "occurred": onehot.ravel()})
    edges = [0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 1.01]
    quoted["bucket"] = pd.cut(quoted.said, edges, right=False)
    reliability = (
        quoted.groupby("bucket", observed=True)
        .agg(n=("occurred", "size"), said=("said", "mean"),
             happened=("occurred", "mean"))
        .query("n >= 30")
        .reset_index(drop=True)
    )
    reliability["gap"] = reliability.happened - reliability.said

    work = frame.assign(
        confidence=model.max(1),
        pick=[config.CLASS_LABELS[i] for i in picked],
        result=[config.CLASS_LABELS[i] for i in truth],
        correct=picked == truth,
        market_correct=market_picked == truth,
        surprise=model[np.arange(len(truth)), truth]
        - market[np.arange(len(truth)), truth],
    )

    by_season = (
        work.groupby("Season")
        .agg(n=("correct", "size"), model=("correct", "mean"),
             market=("market_correct", "mean"))
        .reset_index()
    )

    misses = work.loc[~work.correct].nlargest(8, "confidence")
    best = work.loc[work.correct].nlargest(8, "surprise")

    return Scorecard(
        n=len(work),
        seasons=work.Season.nunique(),
        span=f"{work.Season.min()} – {work.Season.max()}",
        hit=float(work.correct.mean()),
        market_hit=float(work.market_correct.mean()),
        brier=brier(model),
        market_brier=brier(market),
        closer=float((((model - onehot) ** 2).sum(1)
                      < ((market - onehot) ** 2).sum(1)).mean()),
        reliability=reliability,
        by_season=by_season,
        misses=misses,
        best=best,
    )


# --------------------------------------------------------------------------- #
# Charts (hand-built SVG; no runtime dependency)
# --------------------------------------------------------------------------- #


def reliability_chart(table: pd.DataFrame) -> str:
    """Predicted probability against observed frequency, with the ideal line."""
    w, h, pad = 460, 460, 52
    span = w - 2 * pad
    x = lambda v: pad + v * span                      # noqa: E731
    y = lambda v: h - pad - v * span                  # noqa: E731

    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="Predicted probability against observed frequency">'
    ]
    for t in (0, 0.2, 0.4, 0.6, 0.8):
        parts.append(f'<line class="grid" x1="{x(t):.1f}" y1="{y(0):.1f}" '
                     f'x2="{x(t):.1f}" y2="{y(0.85):.1f}"/>')
        parts.append(f'<line class="grid" x1="{x(0):.1f}" y1="{y(t):.1f}" '
                     f'x2="{x(0.85):.1f}" y2="{y(t):.1f}"/>')
        parts.append(f'<text class="tick" x="{x(t):.1f}" y="{y(0) + 18:.1f}" '
                     f'text-anchor="middle">{t:.0%}</text>')
        if t:
            parts.append(f'<text class="tick" x="{x(0) - 10:.1f}" '
                         f'y="{y(t) + 4:.1f}" text-anchor="end">{t:.0%}</text>')

    parts.append(f'<line class="ideal" x1="{x(0):.1f}" y1="{y(0):.1f}" '
                 f'x2="{x(0.85):.1f}" y2="{y(0.85):.1f}"/>')
    parts.append(f'<text class="ideal-label" x="{x(0.60):.1f}" '
                 f'y="{y(0.66):.1f}" transform="rotate(-45 {x(0.60):.1f} '
                 f'{y(0.66):.1f})">perfect calibration</text>')

    biggest = table.n.max()
    for _, row in table.iterrows():
        radius = 5 + 11 * (row.n / biggest) ** 0.5
        label = (f"Said {row.said:.0%} · happened {row.happened:.0%} · "
                 f"{int(row.n):,} predictions")
        parts.append(
            f'<g class="pt"><title>{html.escape(label)}</title>'
            f'<circle cx="{x(row.said):.1f}" cy="{y(row.happened):.1f}" '
            f'r="{radius:.1f}"/></g>'
        )

    parts.append(f'<text class="axis" x="{x(0.42):.1f}" y="{h - 8}" '
                 f'text-anchor="middle">what the model said</text>')
    parts.append(f'<text class="axis" transform="rotate(-90 14 {h / 2:.0f})" '
                 f'x="14" y="{h / 2:.0f}" text-anchor="middle">'
                 f'how often it happened</text>')
    parts.append("</svg>")
    return "".join(parts)


def season_chart(table: pd.DataFrame) -> str:
    """Top-pick hit rate per season, model against the bookmakers."""
    w, h = 720, 300
    left, right, top, bottom = 44, 16, 22, 54
    plot_w, plot_h = w - left - right, h - top - bottom
    lo, hi = 0.40, 0.65
    step = plot_w / (len(table) - 1)
    x = lambda i: left + i * step                                    # noqa: E731
    y = lambda v: top + plot_h - (v - lo) / (hi - lo) * plot_h       # noqa: E731

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Hit rate by season">']
    for t in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        parts.append(f'<line class="grid" x1="{left}" y1="{y(t):.1f}" '
                     f'x2="{w - right}" y2="{y(t):.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 8}" y="{y(t) + 4:.1f}" '
                     f'text-anchor="end">{t:.0%}</text>')

    for key, css in (("market", "mkt"), ("model", "mdl")):
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}"
                       for i, v in enumerate(table[key]))
        parts.append(f'<polyline class="line {css}" points="{pts}"/>')

    for i, row in table.iterrows():
        short = row.Season[2:4] + "/" + row.Season[-2:]
        parts.append(f'<text class="tick" x="{x(i):.1f}" y="{h - 30}" '
                     f'text-anchor="middle">{short}</text>')
        for key, css in (("market", "mkt"), ("model", "mdl")):
            tip = (f"{row.Season} · {'bookmakers' if key == 'market' else 'model'} "
                   f"{row[key]:.1%} of {int(row.n)} matches")
            parts.append(
                f'<g class="pt {css}"><title>{html.escape(tip)}</title>'
                f'<circle cx="{x(i):.1f}" cy="{y(row[key]):.1f}" r="4.5"/></g>'
            )

    last = len(table) - 1
    parts.append(f'<text class="dlabel mdl" x="{x(last) + 8:.1f}" '
                 f'y="{y(table.model.iloc[-1]) + 4:.1f}">model</text>')
    parts.append(f'<text class="dlabel mkt" x="{x(last) + 8:.1f}" '
                 f'y="{y(table.market.iloc[-1]) - 8:.1f}">bookmakers</text>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #


def rows(frame: pd.DataFrame, kind: str) -> str:
    words = {"H": "home win", "D": "draw", "A": "away win"}
    out = []
    for _, r in frame.iterrows():
        score = f"{int(r.FullTimeHomeGoals)}–{int(r.FullTimeAwayGoals)}"
        claim = (f"{r.confidence:.0%} {words[r.pick]}" if kind == "miss"
                 else f"+{r.surprise:.0%} vs market")
        out.append(
            f"<tr><td class='dt'>{r.MatchDate.date()}</td>"
            f"<td class='tm'>{html.escape(r.HomeTeam)}</td>"
            f"<td class='sc'>{score}</td>"
            f"<td class='tm'>{html.escape(r.AwayTeam)}</td>"
            f"<td class='cl'>{claim}</td>"
            f"<td class='rs'>{words[r.result]}</td></tr>"
        )
    return "".join(out)


def build(card: Scorecard) -> str:
    over = card.reliability.loc[card.reliability.gap.idxmin()]
    return TEMPLATE.format(
        n=f"{card.n:,}",
        seasons=card.seasons,
        span=card.span,
        hit=f"{card.hit:.1%}",
        market_hit=f"{card.market_hit:.1%}",
        brier=f"{card.brier:.4f}",
        market_brier=f"{card.market_brier:.4f}",
        closer=f"{card.closer:.0%}",
        reliability=reliability_chart(card.reliability),
        seasons_chart=season_chart(card.by_season),
        misses=rows(card.misses, "miss"),
        best=rows(card.best, "best"),
        over_said=f"{over.said:.0%}",
        over_happened=f"{over.happened:.0%}",
        over_n=f"{int(over.n):,}",
        ink_light=INK_LIGHT, ink_dark=INK_DARK,
        mkt_light=MKT_LIGHT, mkt_dark=MKT_DARK,
    )


def main() -> None:
    card = compute(load())
    OUTPUT.write_text(build(card), encoding="utf-8")
    print(f"Wrote {OUTPUT} — {card.n:,} predictions across {card.seasons} seasons.")
    print(json.dumps(
        {"hit_rate": round(card.hit, 4), "market_hit_rate": round(card.market_hit, 4),
         "brier": round(card.brier, 4), "market_brier": round(card.market_brier, 4),
         "closer_than_market": round(card.closer, 4)}, indent=2))


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Scorecard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=Newsreader:opsz,wght@6..72,400;6..72,500&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  :root {{
    --paper:#F1F3F2; --surface:#FFF; --surface-2:#E6EBE8;
    --ink:#10161B; --ink-2:#38454E; --muted:#6C7A77;
    --rule:#D2D9D6; --grid:#E3E8E6;
    --model:{ink_light}; --market:{mkt_light};
    --f-display:"Archivo","Helvetica Neue",Arial,sans-serif;
    --f-body:"Newsreader",Georgia,serif;
    --f-mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  }}
  @media (prefers-color-scheme:dark) {{
    :root:not([data-theme="light"]) {{
      --paper:#0C1012; --surface:#141A1D; --surface-2:#1C2428;
      --ink:#E9EEEC; --ink-2:#B2BFBB; --muted:#83918D;
      --rule:#283236; --grid:#232D31;
      --model:{ink_dark}; --market:{mkt_dark};
    }}
  }}
  :root[data-theme="dark"] {{
    --paper:#0C1012; --surface:#141A1D; --surface-2:#1C2428;
    --ink:#E9EEEC; --ink-2:#B2BFBB; --muted:#83918D;
    --rule:#283236; --grid:#232D31;
    --model:{ink_dark}; --market:{mkt_dark};
  }}
  *,*::before,*::after {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--paper); color:var(--ink);
         font-family:var(--f-body); font-size:1.0625rem; line-height:1.6;
         -webkit-font-smoothing:antialiased; }}
  .wrap {{ max-width:64rem; margin:0 auto; padding:0 1.75rem 5rem; }}
  h1,h2,.lbl,.tile-v,.dlabel {{ font-family:var(--f-display); }}
  h1 {{ font-size:clamp(2.4rem,6vw,3.9rem); font-weight:800; letter-spacing:-.028em;
        line-height:1; margin:0; text-wrap:balance; }}
  h2 {{ font-size:clamp(1.35rem,2.4vw,1.7rem); font-weight:700;
        letter-spacing:-.016em; margin:0; text-wrap:balance; }}
  p {{ margin:0; max-width:38rem; }}
  .lbl {{ font-family:var(--f-mono); font-size:.68rem; font-weight:600;
          letter-spacing:.13em; text-transform:uppercase; color:var(--muted); }}
  .lede {{ font-size:clamp(1.1rem,1.9vw,1.28rem); line-height:1.5; color:var(--ink-2);
           max-width:34rem; }}
  .note {{ font-size:.92rem; color:var(--muted); line-height:1.5; max-width:38rem; }}
  header {{ border-bottom:2px solid var(--ink); padding:3.5rem 0 2rem;
            display:flex; flex-direction:column; gap:1.4rem; }}
  .kicker {{ display:flex; flex-wrap:wrap; gap:.5rem 1.4rem; }}
  .strip {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(10rem,1fr));
            gap:1px; background:var(--rule); border-bottom:1px solid var(--rule); }}
  .tile {{ background:var(--paper); padding:1.15rem 0 1.35rem;
           display:flex; flex-direction:column; gap:.35rem; }}
  .tile-v {{ font-size:1.7rem; font-weight:700; letter-spacing:-.02em; line-height:1;
             font-variant-numeric:tabular-nums; }}
  .tile-v.mdl {{ color:var(--model); }} .tile-v.mkt {{ color:var(--market); }}
  .tile-n {{ font-size:.88rem; color:var(--muted); line-height:1.35; }}
  section {{ padding-top:3.5rem; }}
  .head {{ display:flex; flex-direction:column; gap:.65rem; padding-bottom:1.5rem; }}
  .fig {{ background:var(--surface); border:1px solid var(--rule); padding:1.25rem;
          overflow-x:auto; }}
  .two {{ display:grid; grid-template-columns:minmax(0,460px) minmax(0,1fr);
          gap:2rem; align-items:center; }}
  @media (max-width:52rem) {{ .two {{ grid-template-columns:1fr; }} }}
  svg {{ display:block; width:100%; height:auto; }}
  .grid {{ stroke:var(--grid); stroke-width:1; }}
  .ideal {{ stroke:var(--muted); stroke-width:1.5; stroke-dasharray:4 4; }}
  .ideal-label,.axis {{ font-family:var(--f-mono); font-size:10px; fill:var(--muted);
                        letter-spacing:.06em; text-transform:uppercase; }}
  .tick {{ font-family:var(--f-mono); font-size:10px; fill:var(--muted);
           font-variant-numeric:tabular-nums; }}
  .pt circle {{ fill:var(--model); stroke:var(--surface); stroke-width:2; }}
  .pt.mkt circle {{ fill:var(--market); }}
  .pt:hover circle {{ stroke:var(--ink); }}
  .line {{ fill:none; stroke-width:2; }}
  .line.mdl {{ stroke:var(--model); }} .line.mkt {{ stroke:var(--market); }}
  .dlabel {{ font-size:11px; font-weight:700; }}
  .dlabel.mdl {{ fill:var(--model); }} .dlabel.mkt {{ fill:var(--market); }}
  .scroll {{ overflow-x:auto; border:1px solid var(--rule); background:var(--surface); }}
  table {{ border-collapse:collapse; width:100%; min-width:36rem; font-size:.93rem; }}
  th,td {{ text-align:left; padding:.6rem .9rem; border-bottom:1px solid var(--rule); }}
  th {{ font-family:var(--f-mono); font-size:.64rem; font-weight:600;
        letter-spacing:.11em; text-transform:uppercase; color:var(--muted);
        background:var(--surface-2); white-space:nowrap; }}
  tbody tr:last-child td {{ border-bottom:none; }}
  td {{ color:var(--ink-2); }}
  td.dt,td.sc,td.cl {{ font-family:var(--f-mono); font-variant-numeric:tabular-nums;
                       white-space:nowrap; }}
  td.tm {{ color:var(--ink); font-weight:500; }}
  td.sc {{ color:var(--ink); font-weight:600; }}
  td.cl {{ color:var(--model); }}
  .stack {{ display:flex; flex-direction:column; gap:1rem; }}
  footer {{ margin-top:3.5rem; padding-top:1.4rem; border-top:1px solid var(--rule); }}
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="kicker">
      <span class="lbl">Premier League Predictor</span>
      <span class="lbl">{span}</span>
      <span class="lbl">{n} predictions</span>
    </div>
    <h1>The scorecard</h1>
    <p class="lede">Every football prediction site tells you what will happen.
    This page keeps score of how often it was wrong.</p>
  </header>

  <div class="strip">
    <div class="tile"><span class="lbl">Predictions</span>
      <span class="tile-v">{n}</span>
      <span class="tile-n">Across {seasons} seasons, none of them seen during training.</span></div>
    <div class="tile"><span class="lbl">Top pick correct</span>
      <span class="tile-v mdl">{hit}</span>
      <span class="tile-n">Bookmakers managed {market_hit} on the same matches.</span></div>
    <div class="tile"><span class="lbl">Brier score</span>
      <span class="tile-v mdl">{brier}</span>
      <span class="tile-n">Bookmakers {market_brier}. Lower is better — they win.</span></div>
    <div class="tile"><span class="lbl">Closer than the market</span>
      <span class="tile-v">{closer}</span>
      <span class="tile-n">Share of individual matches where we were nearer the truth.</span></div>
  </div>

  <section>
    <div class="head">
      <span class="lbl">01 — Do the percentages mean anything</span>
      <h2>When it says 60%, does it happen 60% of the time?</h2>
      <p class="note">Every probability the model ever quoted, pooled and bucketed.
      A dot on the dashed line means that bucket was exactly right. Dot size is how
      many predictions sit in it.</p>
    </div>
    <div class="two">
      <div class="fig">{reliability}</div>
      <div class="stack">
        <p>This is the honest test. Being right often is easy if you only ever
        predict the favourite. Being <em>calibrated</em> means the number itself
        carries information — that a 30% is genuinely rarer than a 60%.</p>
        <p>The dots track the line closely, with one consistent bias: at the
        confident end the model says {over_said} and reality delivers
        {over_happened} across {over_n} predictions. It is slightly too sure of
        its favourites — which is exactly what the worst misses below look like.</p>
      </div>
    </div>
  </section>

  <section>
    <div class="head">
      <span class="lbl">02 — Season by season</span>
      <h2>Never better than the bookmakers for long</h2>
      <p class="note">Share of matches where the top pick was correct. The model
      leads in two seasons out of twelve.</p>
    </div>
    <div class="fig">{seasons_chart}</div>
  </section>

  <section>
    <div class="head">
      <span class="lbl">03 — The failures</span>
      <h2>The eight worst calls</h2>
      <p class="note">Highest-confidence predictions that were wrong. Publishing
      these is the point of the page.</p>
    </div>
    <div class="scroll"><table>
      <thead><tr><th>Date</th><th>Home</th><th>Score</th><th>Away</th>
      <th>We said</th><th>Actually</th></tr></thead>
      <tbody>{misses}</tbody>
    </table></div>
    <p class="note" style="margin-top:1rem">All eight are the same failure: a big
    side not beating a smaller one at home — five draws and three defeats, each
    priced at under 13%. That is the overconfidence in the chart above, made
    concrete. The model has never learned that good teams have flat afternoons.</p>
  </section>

  <section>
    <div class="head">
      <span class="lbl">04 — The wins</span>
      <h2>Where we most disagreed with the market, and were right</h2>
      <p class="note">Correct calls, ranked by how much more probability the model
      gave the true outcome than the bookmakers did.</p>
    </div>
    <div class="scroll"><table>
      <thead><tr><th>Date</th><th>Home</th><th>Score</th><th>Away</th>
      <th>Edge on truth</th><th>Result</th></tr></thead>
      <tbody>{best}</tbody>
    </table></div>
  </section>

  <footer>
    <p class="note">Predictions come from walk-forward evaluation: to forecast
    season N the model is trained only on seasons before N, so no result here was
    available to the model that produced it. Bookmaker probabilities have the
    margin removed before comparison. Regenerate with
    <code>python -m src.scorecard</code>.</p>
  </footer>

</div>
</body>
</html>
"""


if __name__ == "__main__":
    main()
