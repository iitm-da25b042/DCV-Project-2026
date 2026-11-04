"""Static dashboard page (docs/index.html) exported for GitHub Pages.
Run python export_static.py
Tabs, headline metrics, charts, captions and text are preserved, with chart hover and zoom. 
State selectors and chart controls are replaced by fixed chart views.
Run python app.py to ensure full interactivity.
"""

import html
from pathlib import Path

import plotly.io as pio

import data as D
import figures as F

HERE = Path(__file__).resolve().parent
OUT = HERE/"docs"/"index.html"
PLOTLY_JS = "cdn"

b = D.load_all()

TABS = [
    ("1. Spending landscape", "How much does each state spend on health?",
     "Spending per capita in 2019-20, how unevenly it is spread, and whether states spent what they budgeted.",
     [(F.map_spending, False), (F.lollipop_spending, False), (F.rank_shift, False),
      (F.treemap_spending, False), (F.spending_histogram, False), (F.budget_credibility, False)]),
    ("2. Health outcomes", "How healthy are mothers and children?",
     "NFHS-5 health outcomes and care indicators for each state, from child survival to the sex ratio at birth.",
     [(lambda b, s: F.split_heatmap(b, s, "Outcomes"), False), (lambda b, s: F.split_heatmap(b, s, "Care"), False),
      (F.index_scatter, False), (lambda b, s: F.rural_urban(b, s, "imr"), False),
      (F.outcome_spread, False), (F.sex_ratio_birth, False)]),
    ("3. Spending vs outcomes", "Do states that spend more have better outcomes?",
     "The core question, shown for the Outcome Index, for infant mortality, and without the six north-eastern states.",
     [(lambda b, s: F.hero_scatter(b, s, "outcome_idx", "Total", False), True),
      (lambda b, s: F.hero_scatter(b, s, "imr", "Total", False), False),
      (lambda b, s: F.hero_scatter(b, s, "outcome_idx", "Total", True), False),
      (F.correlation_barcode, False), (F.quarter_bars, False), (F.value_for_money, False), (F.two_indices, False)]),
    ("4. Is it really the spending?", "Is it really the spending?",
     "Correlation is not causation. These charts test the main result against other explanations, unusual states, "
     "the timing of the survey, and what families pay.",
     [(lambda b, s: F.confounder_scatter(b, s, "school"), False), (F.confounder_summary, False),
      (F.sensitivity, False), (F.oop_scatter, False)]),
    ("5. Data quality", "What did cleaning change?",
     "The steps from the cleaning notebook: what was wrong in the raw file, where the gaps were, how they were "
     "filled, and which states look unusual.",
     [(F.cleaning_summary, False), (F.missing_by_state, False),
      (lambda b, s: F.imputation_violin(b, s, "Urban"), True), (F.outlier_agreement, True)]),
]
 
first = True
 
def chart_card(func, wide):
    global first
    fig, caption = func(b, None)
    div = pio.to_html(fig, full_html=False, include_plotlyjs=PLOTLY_JS if first else False,
                      config={"displaylogo": False, "responsive": True})
    first = False
    return f"<section class='card{' wide' if wide else ''}'>{div}<p class='caption'>{html.escape(caption)}</p></section>"
 
def kpi_row(tab):
    return "<div class='kpis'>" + "".join(
        f"<div class='kpi{' accent' if k.get('accent') else ''}'><div class='kpi-label'>{html.escape(k['label'])}</div>"
        f"<div class='kpi-value'>{html.escape(str(k['value']))}</div>"
        f"<div class='kpi-note'>{html.escape(k['note'])}</div></div>"
        for k in F.tab_kpis(b, tab)) + "</div>"
 
def intro(question, text):
    return f"<div class='intro'><h2>{html.escape(question)}</h2><p>{html.escape(text)}</p></div>"
 
def findings_tab():
    finds = "".join(
        f"<article class='finding'><div class='finding-n'>{i:02d}</div><div><h3>{html.escape(f['title'])}</h3>"
        f"<p>{html.escape(f['body'])}</p><div class='finding-ref'>{html.escape(f['tab'])}</div></div></article>"
        for i, f in enumerate(F.findings(b), start=1))
 
    def block(title, rows):
        return (f"<section class='card notes'><h2>{title}</h2>"
                + "".join(f"<div><h4>{html.escape(h)}</h4><p>{html.escape(x)}</p></div>" for h, x in rows)
                + "</section>")
 
    return (intro("What we found", "The main conclusions, each with the tab that shows the evidence. "
                                   "All numbers are calculated from the data.")
            + f"<section class='card findings'>{finds}</section>"
            + f"<div class='grid'>{block('How the analysis was done', F.methods(b))}"
            f"{block('Limitations', F.limitations(b))}</div>")
 
buttons, panels = [], []
for i, (label, question, text, charts) in enumerate(TABS):
    cards = "".join(chart_card(f, wide) for f, wide in charts)
    panels.append(f"<div class='tab-content panel' id='p{i}'>{intro(question, text)}{kpi_row(i + 1)}"
                  f"<div class='grid'>{cards}</div></div>")
    buttons.append(f"<button class='tab' onclick='show({i})'>{label}</button>")
panels.append(f"<div class='tab-content panel' id='p{len(TABS)}'>{findings_tab()}</div>")
buttons.append(f"<button class='tab' onclick='show({len(TABS)})'>6. Findings &amp; methods</button>")
 
css = (HERE / "assets" / "style.css").read_text(encoding="utf-8")
page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Health spending &amp; maternal and child health</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@600;700&display=swap" rel="stylesheet">
<style>
{css}
.panel {{ display: none; }}
.panel.active {{ display: flex; }}
.tabs {{ margin-bottom: 4px; }}
.static-note {{ font-size: 13px; color: var(--faint); margin: 6px 0 0; }}
</style></head>
<body><div class="page">
<header class="header"><div class="title-block">
<h1>Does spending more on health buy better maternal and child health?</h1>
<p class="subtitle">{len(b.total)} Indian states · State health expenditure 2019-20 (RBI) and NFHS-5 (2019-21)</p>
<p class="authors">H Sanjana Bhat · Bhavana B · Bolleddula Eshita</p>
<p class="static-note">This is the static version, with hover and zoom working on all charts. For the state selector and chart controls,
run the Dash app (python app.py).</p>
</div></header>
<nav class="tabs">{''.join(buttons)}</nav>
{''.join(panels)}
<footer class="footer"><p>Sources: Reserve Bank of India, State Finances: A Study of Budgets (health expenditure,
2019-20); National Family Health Survey (NFHS-5), 2019-21. Map boundaries: Census 2011 state outlines.</p></footer>
</div>
<script>
function show(i) {{
  document.querySelectorAll('.tab').forEach((t, j) => t.classList.toggle('tab--selected', i === j));
  document.querySelectorAll('.panel').forEach((p, j) => p.classList.toggle('active', i === j));
  // charts drawn in a hidden tab need resizing once the tab is visible
  document.querySelectorAll('#p' + i + ' .plotly-graph-div').forEach(d => Plotly.Plots.resize(d));
}}
show(0);
</script>
</body></html>"""
 
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(page, encoding="utf-8")
print(f"Wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size / 1e6:.1f} MB)")
 