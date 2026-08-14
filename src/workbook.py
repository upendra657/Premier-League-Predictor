"""Export the backtest results as a formula-driven Excel workbook.

The aggregate sheets do not contain baked-in numbers. Every headline figure is
an Excel formula over the raw ``Bet Ledger`` and ``All Candidates`` sheets, so
the reader can re-sort, filter or pivot the underlying rows and watch the
summary move with them -- and check the published figures independently.
"""

from __future__ import annotations

import json

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.table import Table, TableStyleInfo

from src import config
from src.backtest import build_candidates, run_backtest

FONT = "Arial"
INK = "1A1A1A"
MUTED = "6B6B6B"
BLUE = "0000FF"          # hardcoded inputs, per financial-model convention
HEAD_FILL = "1F3864"
BAND_FILL = "DDEBF7"
WARN_FILL = "FCE4E4"

PCT = "0.0%"
PCT2 = "0.00%"
GBP = "£#,##0.00"
GBP0 = "£#,##0"
ODDS = "0.00"
NUM4 = "0.0000"


# --------------------------------------------------------------------------- #
# Styling helpers
# --------------------------------------------------------------------------- #


def _title(sheet, cell: str, text: str, size: int = 14) -> None:
    sheet[cell] = text
    sheet[cell].font = Font(name=FONT, size=size, bold=True, color=INK)


def _note(sheet, cell: str, text: str, italic: bool = True) -> None:
    sheet[cell] = text
    sheet[cell].font = Font(name=FONT, size=9, italic=italic, color=MUTED)
    sheet[cell].alignment = Alignment(wrap_text=True, vertical="top")


def _header_row(sheet, row: int, headers: list[str], start_col: int = 1) -> None:
    thin = Side(style="thin", color="FFFFFF")
    for offset, text in enumerate(headers):
        cell = sheet.cell(row=row, column=start_col + offset, value=text)
        cell.font = Font(name=FONT, size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=HEAD_FILL)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(left=thin, right=thin)


def _widths(sheet, widths: dict[str, int]) -> None:
    for column, width in widths.items():
        sheet.column_dimensions[column].width = width


def _body_font(sheet, min_row: int, max_row: int, max_col: int) -> None:
    for row in sheet.iter_rows(min_row=min_row, max_row=max_row, max_col=max_col):
        for cell in row:
            cell.font = Font(name=FONT, size=10, color=INK)


def _add_table(sheet, name: str, ref: str) -> None:
    table = Table(displayName=name, ref=ref)
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2", showRowStripes=True, showColumnStripes=False
    )
    sheet.add_table(table)


# --------------------------------------------------------------------------- #
# Data sheets
# --------------------------------------------------------------------------- #

LEDGER_COLUMNS = [
    "Season", "Match Date", "Home Team", "Away Team", "Selection",
    "Model Prob", "Market Fair Prob", "Relative Edge", "Decimal Odds",
    "Won", "Unit Return", "Stake (£)", "Profit (£)", "Bankroll After (£)",
    "Edge Band",
]

CANDIDATE_COLUMNS = [
    "Season", "Match Date", "Home Team", "Away Team", "Selection",
    "Model Prob", "Market Fair Prob", "Abs Edge", "Relative Edge",
    "Decimal Odds", "Won", "Unit Return", "Edge Band", "Selected",
]

# Analysis buckets for the edge profile. Deliberately finer than the selection
# band so the shape of the relationship is visible either side of it.
#
# Labels must not begin with "<" or ">": COUNTIFS and friends parse a criterion
# string starting with a comparison operator as an operator rather than as
# literal text, so a band named "< 0%" silently matches nothing and the row
# reads as zero with no error raised.
BAND_LABELS = ["Below 0%", "0-5%", "5-10%", "10-20%", "20-40%", "Above 40%"]

BAND_FORMULA = (
    '=IF({c}{r}<=0,"Below 0%",IF({c}{r}<=0.05,"0-5%",IF({c}{r}<=0.1,"5-10%",'
    'IF({c}{r}<=0.2,"10-20%",IF({c}{r}<=0.4,"20-40%","Above 40%")))))'
)


def _write_ledger(book: Workbook, ledger: pd.DataFrame) -> int:
    sheet = book.create_sheet("Bet Ledger")
    _header_row(sheet, 1, LEDGER_COLUMNS)

    for _, bet in ledger.iterrows():
        sheet.append([
            bet["Season"],
            bet["MatchDate"].to_pydatetime(),
            bet["HomeTeam"],
            bet["AwayTeam"],
            str(bet["outcome"]).title(),
            float(bet["model_prob"]),
            float(bet["fair_prob"]),
            float(bet["relative_edge"]),
            float(bet["odds"]),
            int(bet["won"]),
            float(bet["unit_return"]),
            float(bet["stake"]),
            float(bet["profit"]),
            float(bet["bankroll_after"]),
        ])

    last = len(ledger) + 1
    for row in range(2, last + 1):
        sheet.cell(row=row, column=15, value=BAND_FORMULA.format(c="H", r=row))

    for row in range(2, last + 1):
        sheet.cell(row=row, column=2).number_format = "yyyy-mm-dd"
        for col in (6, 7, 8):
            sheet.cell(row=row, column=col).number_format = PCT
        sheet.cell(row=row, column=9).number_format = ODDS
        sheet.cell(row=row, column=11).number_format = "0.00"
        for col in (12, 13, 14):
            sheet.cell(row=row, column=col).number_format = GBP

    _body_font(sheet, 2, last, 15)
    _widths(sheet, {"A": 9, "B": 12, "C": 15, "D": 15, "E": 10, "F": 11, "G": 15,
                    "H": 12, "I": 12, "J": 7, "K": 11, "L": 11, "M": 11,
                    "N": 17, "O": 11})
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:O{last}"
    return last


def _write_candidates(book: Workbook, candidates: pd.DataFrame) -> int:
    sheet = book.create_sheet("All Candidates")
    _header_row(sheet, 1, CANDIDATE_COLUMNS)

    for _, row_data in candidates.iterrows():
        sheet.append([
            row_data["Season"],
            row_data["MatchDate"].to_pydatetime(),
            row_data["HomeTeam"],
            row_data["AwayTeam"],
            str(row_data["outcome"]).title(),
            float(row_data["model_prob"]),
            float(row_data["fair_prob"]),
            float(row_data["edge"]),
            float(row_data["relative_edge"]),
            float(row_data["odds"]),
            int(row_data["won"]),
            float(row_data["unit_return"]),
        ])

    last = len(candidates) + 1
    for row in range(2, last + 1):
        sheet.cell(row=row, column=13, value=BAND_FORMULA.format(c="I", r=row))
        # The full selection rule, driven by the levers on the Summary sheet.
        # Every clause in src.backtest.select_bets is reproduced here, including
        # the absolute-edge floor and the expected-value test, so that the
        # "Yes" count reconciles exactly with the Bet Ledger row count.
        sheet.cell(
            row=row,
            column=14,
            value=(
                f"=IF(AND(H{row}>=Summary!$B$25,I{row}>=Summary!$B$26,"
                f"I{row}<=Summary!$B$27,J{row}>=Summary!$B$28,"
                f"J{row}<=Summary!$B$29,F{row}*J{row}>1),\"Yes\",\"No\")"
            ),
        )

    for row in range(2, last + 1):
        sheet.cell(row=row, column=2).number_format = "yyyy-mm-dd"
        for col in (6, 7, 8, 9):
            sheet.cell(row=row, column=col).number_format = PCT
        sheet.cell(row=row, column=10).number_format = ODDS
        sheet.cell(row=row, column=12).number_format = "0.00"

    _body_font(sheet, 2, last, 14)
    _widths(sheet, {"A": 9, "B": 12, "C": 15, "D": 15, "E": 10, "F": 11,
                    "G": 15, "H": 10, "I": 12, "J": 12, "K": 7, "L": 11,
                    "M": 11, "N": 10})
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:N{last}"
    return last


# --------------------------------------------------------------------------- #
# Aggregate sheets -- all formulas, no baked-in values
# --------------------------------------------------------------------------- #


def _write_summary(
    book: Workbook, results: dict, ledger_last: int, cand_last: int
) -> None:
    sheet = book.create_sheet("Summary", 1)
    back = results["backtest"]
    data = results["dataset"]
    led = "'Bet Ledger'"

    _title(sheet, "A1", "Premier League Decision Engine — Results Summary")
    _note(
        sheet, "A2",
        "Black figures are live Excel formulas over the Bet Ledger and All Candidates "
        "sheets. Blue figures are inputs or values carried from reports/results.json. "
        "Filter the raw sheets and the black figures recalculate.",
    )

    _title(sheet, "A4", "Headline — line-shopped, 12 held-out seasons", 11)
    _header_row(sheet, 5, ["Metric", "Value", "How it is computed"])

    rows = [
        ("Bets placed", f"=COUNTA({led}!A2:A{ledger_last})", "Rows in Bet Ledger"),
        ("Hit rate", f"=AVERAGE({led}!J2:J{ledger_last})", "Mean of Won (1/0)", PCT),
        ("Average odds", f"=AVERAGE({led}!I2:I{ledger_last})", "Mean decimal odds", ODDS),
        ("Flat-stake ROI", f"=AVERAGE({led}!K2:K{ledger_last})",
         "Mean unit return — ROI per £1 risked at constant stake", PCT2),
        ("Total staked (Kelly)", f"=SUM({led}!L2:L{ledger_last})",
         "Sum of stakes, compounding", GBP0),
        ("Total profit (Kelly)", f"=SUM({led}!M2:M{ledger_last})", "Sum of profit", GBP0),
        ("ROI on turnover (Kelly)",
         f"=SUM({led}!M2:M{ledger_last})/SUM({led}!L2:L{ledger_last})",
         "Profit ÷ turnover", PCT2),
        ("Final bankroll", f"={led}!N{ledger_last}",
         "Last row of Bankroll After", GBP0),
    ]
    for index, item in enumerate(rows):
        row = 6 + index
        sheet.cell(row=row, column=1, value=item[0]).font = Font(name=FONT, size=10, color=INK)
        value_cell = sheet.cell(row=row, column=2, value=item[1])
        value_cell.font = Font(name=FONT, size=10, bold=True, color=INK)
        if len(item) > 3:
            value_cell.number_format = item[3]
        _note(sheet, f"C{row}", item[2], italic=False)

    _title(sheet, "A15", "The three ROI scenarios", 11)
    _note(
        sheet, "A16",
        "Only the first row is derived from this workbook's ledger. The other two come "
        "from reports/results.json and are shown in blue because nothing here recomputes them.",
    )
    _header_row(sheet, 17, ["Scenario", "Bets", "Flat ROI", "95% CI low",
                            "95% CI high", "p(ROI≤0)"])

    full = back["full_period"]
    sheet.cell(row=18, column=1, value="Line-shopped, full period")
    sheet.cell(row=18, column=2, value=f"=COUNTA({led}!A2:A{ledger_last})")
    sheet.cell(row=18, column=3, value=f"=AVERAGE({led}!K2:K{ledger_last})")
    for col, key in ((4, "roi_ci_low"), (5, "roi_ci_high"), (6, "p_roi_not_positive")):
        cell = sheet.cell(row=18, column=col, value=full[key])
        cell.font = Font(name=FONT, size=10, color=BLUE)

    for offset, (label, block) in enumerate(
        (("Single bookmaker", back["single_book"]), ("Strict holdout", back["holdout"]))
    ):
        row = 19 + offset
        sheet.cell(row=row, column=1, value=label)
        for col, key in ((2, "n_bets"), (3, "flat_stake_roi"), (4, "roi_ci_low"),
                         (5, "roi_ci_high"), (6, "p_roi_not_positive")):
            cell = sheet.cell(row=row, column=col, value=block[key])
            cell.font = Font(name=FONT, size=10, color=BLUE)

    for row in range(18, 21):
        for col in (3, 4, 5):
            sheet.cell(row=row, column=col).number_format = PCT2
        sheet.cell(row=row, column=6).number_format = "0.000"
        sheet.cell(row=row, column=1).font = Font(name=FONT, size=10, color=INK)

    for col in range(1, 7):
        sheet.cell(row=21, column=col).fill = PatternFill("solid", fgColor=WARN_FILL)
    sheet.cell(
        row=21, column=1,
        value="Read the holdout row before quoting the headline: refitting the selection "
              "band on 2013/14–2018/19 and applying it blind drops ROI to +1.5%, with a "
              "confidence interval straddling zero.",
    ).font = Font(name=FONT, size=9, bold=True, color="B00000")
    sheet.merge_cells("A21:F21")
    sheet.row_dimensions[21].height = 28
    sheet["A21"].alignment = Alignment(wrap_text=True, vertical="center")

    _title(sheet, "A24", "Selection levers — pre-registered band", 11)
    strategy = config.BACKTEST_CONFIG.strategy
    levers = [
        ("Min absolute edge", strategy.min_edge, PCT,
         "Model probability must exceed the fair price by this margin"),
        ("Min relative edge", strategy.min_relative_edge, PCT,
         "Below this the signal is noise (see Edge Profile)"),
        ("Max relative edge", strategy.max_relative_edge, PCT,
         "Above this the model is overconfident on longshots"),
        ("Min decimal odds", strategy.min_odds, ODDS,
         "Avoids heavy favourites where the vig dominates"),
        ("Max decimal odds", strategy.max_odds, ODDS, "Caps longshot exposure"),
        ("Kelly fraction", config.BACKTEST_CONFIG.kelly_fraction, PCT,
         "Quarter-Kelly, to survive estimation error"),
        ("Max daily exposure", config.BACKTEST_CONFIG.max_daily_exposure, PCT,
         "Cap on combined stake per matchday"),
        ("Starting bankroll", config.BACKTEST_CONFIG.starting_bankroll, GBP0, "Simulation start"),
    ]
    for index, (label, value, fmt, why) in enumerate(levers):
        row = 25 + index
        sheet.cell(row=row, column=1, value=label).font = Font(name=FONT, size=10, color=INK)
        cell = sheet.cell(row=row, column=2, value=value)
        cell.font = Font(name=FONT, size=10, bold=True, color=BLUE)
        cell.number_format = fmt
        cell.fill = PatternFill("solid", fgColor=BAND_FILL)
        _note(sheet, f"C{row}", why, italic=False)

    sheet.cell(row=33, column=1, value="Candidates selected by these levers").font = Font(
        name=FONT, size=10, color=INK
    )
    check = sheet.cell(
        row=33, column=2,
        value=f"=COUNTIFS('All Candidates'!$N$2:$N${cand_last},\"Yes\")",
    )
    check.font = Font(name=FONT, size=10, bold=True, color=INK)
    _note(
        sheet, "C33",
        "Reconciliation check: must equal the Bet Ledger row count in B6.",
        italic=False,
    )

    _note(
        sheet, "A35",
        "Blue cells are inputs. Editing B25:B29 changes the Selected column on "
        "All Candidates, so you can test a different band against the same data. "
        "It does not re-run the Kelly simulation, whose stakes and bankroll are "
        "fixed at the band above. Note this is the pre-registered band used for the "
        "headline result — the strict-holdout row on this sheet uses a different, "
        "refitted band (min 10%, max 30%).",
    )

    _title(sheet, "A38", "Dataset", 11)
    for index, (label, value) in enumerate([
        ("Fixtures", data["fixtures"]),
        ("Seasons", data["seasons"]),
        ("Modellable after burn-in", data["modellable"]),
        ("Odds coverage", data["odds_coverage"]),
        ("Candidate selections", f"=COUNTA('All Candidates'!A2:A{cand_last})"),
    ]):
        row = 39 + index
        sheet.cell(row=row, column=1, value=label).font = Font(name=FONT, size=10, color=INK)
        cell = sheet.cell(row=row, column=2, value=value)
        cell.font = Font(name=FONT, size=10, bold=True,
                         color=INK if isinstance(value, str) else BLUE)
        if label == "Odds coverage":
            cell.number_format = PCT

    _widths(sheet, {"A": 27, "B": 15, "C": 52, "D": 13, "E": 13, "F": 12})


def _write_edge_profile(book: Workbook, cand_last: int) -> None:
    sheet = book.create_sheet("Edge Profile")
    ref = "'All Candidates'"

    _title(sheet, "A1", "Realised return by relative edge")
    _note(
        sheet, "A2",
        "Every cell is a COUNTIFS/SUMIFS over All Candidates — the whole candidate set, "
        "not just the bets that were placed. This is the finding the strategy is built on: "
        "returns are positive only in the middle bands.",
    )
    _header_row(sheet, 4, ["Edge Band", "Candidates", "Wins", "Hit Rate",
                           "Avg Odds", "Flat ROI"])

    for index, label in enumerate(BAND_LABELS):
        row = 5 + index
        band = f'"{label}"'
        sheet.cell(row=row, column=1, value=label).font = Font(name=FONT, size=10, color=INK)
        sheet.cell(row=row, column=2,
                   value=f"=COUNTIFS({ref}!$M$2:$M${cand_last},{band})")
        sheet.cell(row=row, column=3,
                   value=f"=SUMIFS({ref}!$K$2:$K${cand_last},{ref}!$M$2:$M${cand_last},{band})")
        sheet.cell(row=row, column=4, value=f"=IFERROR(C{row}/B{row},0)")
        sheet.cell(row=row, column=5,
                   value=f"=IFERROR(AVERAGEIFS({ref}!$J$2:$J${cand_last},"
                         f"{ref}!$M$2:$M${cand_last},{band}),0)")
        sheet.cell(row=row, column=6,
                   value=f"=IFERROR(AVERAGEIFS({ref}!$L$2:$L${cand_last},"
                         f"{ref}!$M$2:$M${cand_last},{band}),0)")

        sheet.cell(row=row, column=4).number_format = PCT
        sheet.cell(row=row, column=5).number_format = ODDS
        sheet.cell(row=row, column=6).number_format = PCT2
        if label in ("10-20%", "20-40%"):
            for col in range(1, 7):
                sheet.cell(row=row, column=col).fill = PatternFill("solid", fgColor=BAND_FILL)

    sheet.cell(row=11, column=1, value="Total").font = Font(name=FONT, size=10, bold=True)
    sheet.cell(row=11, column=2, value="=SUM(B5:B10)").font = Font(name=FONT, size=10, bold=True)
    sheet.cell(row=11, column=3, value="=SUM(C5:C10)").font = Font(name=FONT, size=10, bold=True)

    _note(
        sheet, "A13",
        "Shaded rows are the selected band. Bands are cut at 0/5/10/20/40% relative edge "
        "by the Edge Band formula in All Candidates column M; edit that formula to re-cut them.",
    )
    _body_font(sheet, 5, 10, 6)
    _widths(sheet, {"A": 14, "B": 13, "C": 10, "D": 11, "E": 11, "F": 12})


def _write_equity(book: Workbook, equity: pd.DataFrame, ledger_last: int) -> None:
    sheet = book.create_sheet("Equity by Season")
    led = "'Bet Ledger'"

    _title(sheet, "A1", "Bankroll progression by season")
    _note(
        sheet, "A2",
        "SUMIFS over the Bet Ledger by season. Six of twelve seasons lose money — "
        "the edge is real but far from monotonic.",
    )
    _header_row(sheet, 4, ["Season", "Bets", "Hit Rate", "Staked (£)",
                           "Profit (£)", "Season ROI", "Cumulative Profit (£)"])

    seasons = equity["Season"].tolist()
    for index, season in enumerate(seasons):
        row = 5 + index
        key = f'"{season}"'
        sheet.cell(row=row, column=1, value=season).font = Font(name=FONT, size=10, color=INK)
        sheet.cell(row=row, column=2, value=f"=COUNTIFS({led}!$A$2:$A${ledger_last},{key})")
        sheet.cell(row=row, column=3,
                   value=f"=IFERROR(AVERAGEIFS({led}!$J$2:$J${ledger_last},"
                         f"{led}!$A$2:$A${ledger_last},{key}),0)")
        sheet.cell(row=row, column=4,
                   value=f"=SUMIFS({led}!$L$2:$L${ledger_last},{led}!$A$2:$A${ledger_last},{key})")
        sheet.cell(row=row, column=5,
                   value=f"=SUMIFS({led}!$M$2:$M${ledger_last},{led}!$A$2:$A${ledger_last},{key})")
        sheet.cell(row=row, column=6, value=f"=IFERROR(E{row}/D{row},0)")
        sheet.cell(row=row, column=7, value=f"=SUM($E$5:E{row})")

        sheet.cell(row=row, column=3).number_format = PCT
        for col in (4, 5, 7):
            sheet.cell(row=row, column=col).number_format = GBP
        sheet.cell(row=row, column=6).number_format = PCT2

    total = 5 + len(seasons)
    sheet.cell(row=total, column=1, value="Total").font = Font(name=FONT, size=10, bold=True)
    for col, letter in ((2, "B"), (4, "D"), (5, "E")):
        cell = sheet.cell(row=total, column=col,
                          value=f"=SUM({letter}5:{letter}{total - 1})")
        cell.font = Font(name=FONT, size=10, bold=True)
    sheet.cell(row=total, column=4).number_format = GBP
    sheet.cell(row=total, column=5).number_format = GBP
    cell = sheet.cell(row=total, column=6, value=f"=IFERROR(E{total}/D{total},0)")
    cell.font = Font(name=FONT, size=10, bold=True)
    cell.number_format = PCT2

    _body_font(sheet, 5, total - 1, 7)
    _widths(sheet, {"A": 10, "B": 8, "C": 11, "D": 13, "E": 13, "F": 12, "G": 20})


def _write_metrics(book: Workbook, results: dict) -> None:
    sheet = book.create_sheet("Model Metrics")

    _title(sheet, "A1", "Walk-forward model comparison")
    _note(
        sheet, "A2",
        "4,509 fixtures across 12 held-out seasons; the model is refit each season and "
        "never sees the season it predicts. Lower is better for Brier, log loss, RPS and ECE. "
        "Values are from reports/model_metrics.csv and shown in blue: this workbook does not "
        "re-run training.",
    )
    _header_row(sheet, 4, ["Model", "N", "Brier", "Log Loss", "RPS", "ECE", "Accuracy"])

    for index, metric in enumerate(results["walk_forward_metrics"]):
        row = 5 + index
        is_market = metric["model"] == "market_baseline"
        name = "Market (de-vigged)" if is_market else metric["model"].replace("_", " ")
        cell = sheet.cell(row=row, column=1, value=name)
        cell.font = Font(name=FONT, size=10, bold=is_market, color=INK)
        for col, key, fmt in (
            (2, "n", "#,##0"), (3, "brier", NUM4), (4, "log_loss", NUM4),
            (5, "rps", NUM4), (6, "ece", NUM4), (7, "accuracy", PCT),
        ):
            value_cell = sheet.cell(row=row, column=col, value=metric[key])
            value_cell.font = Font(name=FONT, size=10, bold=is_market, color=BLUE)
            value_cell.number_format = fmt
        if is_market:
            for col in range(1, 8):
                sheet.cell(row=row, column=col).fill = PatternFill("solid", fgColor=BAND_FILL)

    _title(sheet, "A13", "What calibration bought", 11)
    _header_row(sheet, 14, ["Estimator", "Brier Before", "Brier After", "Brier Δ",
                            "ECE Before", "ECE After", "ECE Δ"])
    for index, gain in enumerate(results["calibration_gain"]):
        row = 15 + index
        sheet.cell(row=row, column=1,
                   value=gain["estimator"].replace("_", " ")).font = Font(name=FONT, size=10, color=INK)
        for col, key in ((2, "brier_before"), (3, "brier_after"),
                         (5, "ece_before"), (6, "ece_after")):
            cell = sheet.cell(row=row, column=col, value=gain[key])
            cell.font = Font(name=FONT, size=10, color=BLUE)
            cell.number_format = NUM4
        for col, before, after in ((4, "B", "C"), (7, "E", "F")):
            cell = sheet.cell(row=row, column=col,
                              value=f"=IFERROR(({after}{row}-{before}{row})/{before}{row},0)")
            cell.font = Font(name=FONT, size=10, color=INK)
            cell.number_format = PCT

    _note(
        sheet, "A18",
        "Isotonic regression cut Random Forest's calibration error by more than half. "
        "It did nothing for XGBoost, whose mlogloss objective is already a strictly proper "
        "scoring rule — so the model arrives calibrated and post-hoc fitting only adds noise. "
        "The Δ columns are formulas over the blue cells.",
    )
    _widths(sheet, {"A": 22, "B": 13, "C": 13, "D": 12, "E": 12, "F": 12, "G": 12})


def _write_readme(book: Workbook, results: dict) -> None:
    sheet = book.create_sheet("Read Me", 0)
    data = results["dataset"]

    _title(sheet, "A1", "Premier League Decision Engine — Results Workbook", 16)
    sheet["A2"] = (
        "Calibrated match-outcome probabilities for the English Premier League, and a "
        "backtest of whether acting on them beats the bookmaker."
    )
    sheet["A2"].font = Font(name=FONT, size=11, color=MUTED)

    _title(sheet, "A4", "Sheets", 12)
    guide = [
        ("Summary", "Headline figures, the three ROI scenarios, and the selection levers."),
        ("Model Metrics", "Walk-forward Brier / log loss / RPS / ECE, and the calibration gain."),
        ("Edge Profile", "Return by relative-edge band. The core finding."),
        ("Equity by Season", "Per-season bets, staked, profit and ROI."),
        ("Bet Ledger", f"All {results['backtest']['full_period']['n_bets']:,} placed bets, one row each. Pivot this."),
        ("All Candidates", "Every EV+ opportunity considered, including those rejected."),
    ]
    _header_row(sheet, 5, ["Sheet", "Contents"])
    for index, (name, description) in enumerate(guide):
        row = 6 + index
        sheet.cell(row=row, column=1, value=name).font = Font(name=FONT, size=10, bold=True, color=INK)
        sheet.cell(row=row, column=2, value=description).font = Font(name=FONT, size=10, color=INK)

    _title(sheet, "A14", "How to read the colours", 12)
    for index, (swatch, meaning) in enumerate([
        ("Black", "Live Excel formula over the raw sheets. Filter the data and it recalculates."),
        ("Blue", "A hardcoded input, or a value carried from reports/results.json."),
        ("Shaded", "The selected edge band, or the market baseline row."),
    ]):
        row = 15 + index
        cell = sheet.cell(row=row, column=1, value=swatch)
        cell.font = Font(name=FONT, size=10, bold=True,
                         color=BLUE if swatch == "Blue" else INK)
        if swatch == "Shaded":
            cell.fill = PatternFill("solid", fgColor=BAND_FILL)
        sheet.cell(row=row, column=2, value=meaning).font = Font(name=FONT, size=10, color=INK)

    _title(sheet, "A20", "The one caveat that matters", 12)
    cell = sheet.cell(
        row=21, column=1,
        value=(
            "The headline +7.2% ROI uses a selection band chosen with hindsight over the "
            "whole sample. Refit that band on 2013/14–2018/19 and apply it blind to the "
            "following six seasons and ROI falls to +1.5%, with a 95% confidence interval "
            "that straddles zero. The honest reading is a marginal, unstable edge in a "
            "near-efficient market — not a money printer. Both numbers are on the Summary sheet."
        ),
    )
    cell.font = Font(name=FONT, size=10, color="B00000")
    cell.alignment = Alignment(wrap_text=True, vertical="top")
    sheet.merge_cells("A21:B24")
    for col in ("A", "B"):
        for row in range(21, 25):
            sheet[f"{col}{row}"].fill = PatternFill("solid", fgColor=WARN_FILL)
    # Merged cells do not auto-fit, so the wrapped paragraph needs the height
    # reserved explicitly or it renders clipped.
    for row in range(21, 25):
        sheet.row_dimensions[row].height = 16

    _title(sheet, "A26", "Provenance", 12)
    provenance = [
        ("Source dataset", f"epl_final.csv — {data['fixtures']:,} fixtures, {data['seasons']} seasons (2000/01–2024/25)"),
        ("Odds", f"football-data.co.uk closing odds, {data['odds_coverage']:.1%} coverage"),
        ("Evaluation", "Walk-forward: refit each season, predict the next, never look back"),
        ("Model", "XGBoost, uncalibrated (best Brier of the tree models)"),
        ("Bootstrap", f"{config.BACKTEST_CONFIG.bootstrap_samples:,} resamples, clustered by season"),
        ("Reproduce", "python -m src.train && python -m src.backtest && python -m src.report"),
        ("Generated by", "python -m src.workbook"),
    ]
    for index, (label, value) in enumerate(provenance):
        row = 27 + index
        sheet.cell(row=row, column=1, value=label).font = Font(name=FONT, size=10, bold=True, color=INK)
        sheet.cell(row=row, column=2, value=value).font = Font(name=FONT, size=10, color=MUTED)

    _widths(sheet, {"A": 22, "B": 88})


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> str:
    """Rebuild the results workbook from the stored predictions."""
    results = json.loads((config.REPORT_DIR / "results.json").read_text())
    predictions = pd.read_parquet(config.PROCESSED_DIR / "oos_predictions.parquet")
    market = pd.read_parquet(config.FEATURE_STORE_FILE)

    best = predictions[predictions["variant"] == "xgb_uncalibrated"]
    candidates = build_candidates(best, market, "best_odds")
    full = run_backtest(best, market, odds_source="best_odds", label="workbook")

    book = Workbook()
    book.remove(book.active)

    _write_readme(book, results)
    ledger_last = _write_ledger(book, full.bets)
    cand_last = _write_candidates(book, candidates)
    _write_summary(book, results, ledger_last, cand_last)
    _write_metrics(book, results)
    _write_edge_profile(book, cand_last)
    _write_equity(book, full.equity_curve, ledger_last)

    book._sheets = [
        book["Read Me"], book["Summary"], book["Model Metrics"],
        book["Edge Profile"], book["Equity by Season"],
        book["Bet Ledger"], book["All Candidates"],
    ]

    output = config.REPORT_DIR / "epl_decision_engine_results.xlsx"
    book.save(output)
    return str(output)


if __name__ == "__main__":  # pragma: no cover
    print(main())
