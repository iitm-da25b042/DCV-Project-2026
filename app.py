"""Plotly Dash dashboard: do states that spend more on health have better maternal and child health?

Run:  python app.py   then open http://127.0.0.1:8050
"""

import logging

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

import data as D
import figures as F

B = D.load_all()
STATES = sorted(B.total.State)

GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 2},
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}

# Options for the chart controls
HERO_KEYS = ["outcome_idx", "care_idx"] + [
    i.key for i in D.INDICATORS if i.group in ("Outcome", "Care", "Cost & cover") and B.total[i.key].notna().sum() >= 10
]
HERO_OPTS = [{"label": F.label_any(k), "value": k} for k in HERO_KEYS]
AREA_OPTS = [{"label": a, "value": a} for a in D.AREAS]
UR_OPTS = [
    {"label": D.IND[k].label, "value": k}
    for k in ["imr", "nmr", "u5mr", "stunt", "wast", "underwt", "anaem_child", "anaem_preg", "inst", "anc4", "vacc"]
]
CONF_OPTS = [{"label": F.CONFOUNDER_SHORT[k].capitalize(), "value": k} for k in F.confounders(B)]


# Building blocks
def dropdown(id_, options, value, label, width=240):
    return html.Div(className="control", children=[
        html.Label(label, htmlFor=id_, className="control-label"),
        dcc.Dropdown(id=id_, options=options, value=value, clearable=False, style={"minWidth": f"{width}px"},
                     persistence=True, persistence_type="session"),
    ])


def radio(id_, options, value, label):
    return html.Div(className="control", children=[
        html.Span(label, className="control-label"),
        dcc.RadioItems(id=id_, options=options, value=value, inline=True, className="radio",
                       inputClassName="radio-input", labelClassName="radio-label",
                       persistence=True, persistence_type="session"),
    ])


def card(key, wide=False, controls=None):
    """One chart card: optional controls, the graph and its caption."""
    return html.Section(className="card wide" if wide else "card", children=[
        html.Div(controls, className="controls") if controls else None,
        dcc.Loading(type="dot", color="#2EE6FF",
                    children=dcc.Graph(id=f"g-{key}", config=GRAPH_CONFIG, style={"height": "560px"})),
        html.P(id=f"c-{key}", className="caption"),
    ])


def kpi_row(tab):
    return html.Div(className="kpis", children=[
        html.Div(className="kpi accent" if k.get("accent") else "kpi", children=[
            html.Div(k["label"], className="kpi-label"),
            html.Div(k["value"], className="kpi-value"),
            html.Div(k["note"], className="kpi-note"),
        ]) for k in F.tab_kpis(B, tab)
    ])


def intro(question, text):
    return html.Div(className="intro", children=[html.H2(question), html.P(text)])


# Tabs
def tab_spending():
    return [
        intro("How much does each state spend on health?",
              "Spending per capita in 2019-20, how unevenly it is spread, and whether states spent what they budgeted."),
        kpi_row(1),
        html.Div(className="grid", children=[
            card("map"), card("lollipop"), card("rankshift"), card("treemap"), card("hist"), card("budget"),
        ]),
    ]


def tab_outcomes():
    return [
        intro("How healthy are mothers and children?",
              "NFHS-5 health outcomes and care indicators for each state, from child survival to the sex ratio at birth."),
        kpi_row(2),
        html.Div(className="grid", children=[
            card("heat-out"), card("heat-care"), card("index"),
            card("ruralurban", controls=[dropdown("ur-ind", UR_OPTS, "imr", "Indicator")]),
            card("spread"), card("srb"),
        ]),
    ]


def tab_relationship():
    return [
        intro("Do states that spend more have better outcomes?",
              "The core question. Pick an indicator, switch between Total, Urban and Rural figures, or leave out the "
              "six north-eastern states to see how the link changes."),
        kpi_row(3),
        html.Div(className="grid", children=[
            card("hero", wide=True, controls=[
                dropdown("hero-ind", HERO_OPTS, "outcome_idx", "Indicator", 300),
                radio("hero-area", AREA_OPTS, "Total", "Figures"),
                html.Div(className="control", children=[
                    html.Span("Robustness", className="control-label"),
                    dcc.Checklist(id="hero-drop", options=[{"label": "Leave out the 6 north-eastern states",
                                                            "value": "drop"}],
                                  value=[], className="radio", inputClassName="radio-input",
                                  labelClassName="radio-label"),
                ]),
            ]),
            card("barcode"), card("quarters"), card("value"), card("two-idx"),
        ]),
    ]


def tab_robustness():
    return [
        intro("Is it really the spending?",
              "Correlation is not causation. These charts test the main result against other explanations, unusual "
              "states, the timing of the survey, and what families pay."),
        kpi_row(4),
        html.Div(className="grid", children=[
            card("confounder", controls=[dropdown("conf", CONF_OPTS, "school", "Allow for")]),
            card("conf-sum"), card("sensitivity"), card("oop"),
        ]),
    ]


def tab_quality():
    return [
        intro("What did cleaning change?",
              "The steps from the cleaning notebook: what was wrong in the raw file, where the gaps were, how they "
              "were filled, and which states look unusual."),
        kpi_row(5),
        html.Div(className="grid", children=[
            card("cleaning"), card("missing"),
            card("violin", wide=True, controls=[radio("violin-area", AREA_OPTS, "Urban", "Figures")]),
            card("outliers", wide=True),
        ]),
    ]


def tab_findings():
    finds = [
        html.Article(className="finding", children=[
            html.Div(f"{i:02d}", className="finding-n"),
            html.Div([html.H3(f["title"]), html.P(f["body"]), html.Div(f["tab"], className="finding-ref")]),
        ]) for i, f in enumerate(F.findings(B), start=1)
    ]

    def block(title, rows):
        return html.Section(className="card notes", children=[html.H2(title)] + [
            html.Div([html.H4(h), html.P(x)]) for h, x in rows])

    return [
        intro("What we found", "The main conclusions, each with the tab that shows the evidence. "
                               "All numbers are calculated from the data."),
        html.Section(className="card findings", children=finds),
        html.Div(className="grid", children=[
            block("How the analysis was done", F.methods(B)),
            block("Limitations", F.limitations(B)),
        ]),
    ]


TABS = [
    ("spend", "1. Spending landscape", tab_spending),
    ("outcomes", "2. Health outcomes", tab_outcomes),
    ("relation", "3. Spending vs outcomes", tab_relationship),
    ("robust", "4. Is it really the spending?", tab_robustness),
    ("quality", "5. Data quality", tab_quality),
    ("findings", "6. Findings & methods", tab_findings),
]

# App and layout
app = Dash(
    __name__,
    title="Health spending & maternal and child health",
    external_stylesheets=["https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600"
                          "&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@600;700&display=swap"],
)
server = app.server

app.layout = html.Div(className="page", children=[
    html.Header(className="header", children=[
        html.Div(className="title-block", children=[
            html.H1("Does spending more on health buy better maternal and child health?"),
            html.P(f"{len(B.total)} Indian states · State health expenditure 2019-20 (RBI) and NFHS-5 (2019-21)",
                   className="subtitle"),
            html.P("H Sanjana Bhat · Bhavana B · Bolleddula Eshita",
                   className="authors"),
            ]),
        html.Div(className="control", children=[
            html.Label("Highlight a state", htmlFor="state", className="control-label"),
            dcc.Dropdown(id="state", options=STATES, value=None, clearable=True, placeholder="Choose a state",
                         style={"minWidth": "240px"}, persistence=True, persistence_type="session"),
            html.Span("Or click a state on the map or on most charts", className="hint"),
        ]),
    ]),
    dcc.Tabs(id="tabs", value="spend", className="tabs", parent_className="tabs-parent", children=[
        dcc.Tab(label=label, value=value, className="tab", selected_className="tab--selected",
                children=html.Div(className="tab-content", children=build()))
        for value, label, build in TABS
    ]),
    html.Footer(className="footer", children=html.P(
        "Sources: Reserve Bank of India, State Finances: A Study of Budgets (health expenditure, 2019-20); "
        "National Family Health Survey (NFHS-5), 2019-21. Map boundaries: Census 2011 state outlines.")),
])


# Callbacks: one per chart. Each chart key maps to its function and any controls it listens to.
CHARTS = {
    "map": (F.map_spending, []),
    "lollipop": (F.lollipop_spending, []),
    "rankshift": (F.rank_shift, []),
    "treemap": (F.treemap_spending, []),
    "hist": (F.spending_histogram, []),
    "budget": (F.budget_credibility, []),
    "heat-out": (lambda b, s: F.split_heatmap(b, s, "Outcomes"), []),
    "heat-care": (lambda b, s: F.split_heatmap(b, s, "Care"), []),
    "index": (F.index_scatter, []),
    "ruralurban": (lambda b, s, ind: F.rural_urban(b, s, ind or "imr"), ["ur-ind"]),
    "spread": (F.outcome_spread, []),
    "srb": (F.sex_ratio_birth, []),
    "hero": (lambda b, s, ind, area, drop: F.hero_scatter(b, s, ind or "outcome_idx", area or "Total", bool(drop)),
             ["hero-ind", "hero-area", "hero-drop"]),
    "barcode": (F.correlation_barcode, []),
    "quarters": (F.quarter_bars, []),
    "value": (F.value_for_money, []),
    "two-idx": (F.two_indices, []),
    "confounder": (lambda b, s, conf: F.confounder_scatter(b, s, conf or "school"), ["conf"]),
    "conf-sum": (F.confounder_summary, []),
    "sensitivity": (F.sensitivity, []),
    "oop": (F.oop_scatter, []),
    "cleaning": (F.cleaning_summary, []),
    "missing": (F.missing_by_state, []),
    "violin": (lambda b, s, area: F.imputation_violin(b, s, area or "Urban"), ["violin-area"]),
    "outliers": (F.outlier_agreement, []),
}


def _register(key, func, controls):
    @app.callback(
        Output(f"g-{key}", "figure"),
        Output(f"c-{key}", "children"),
        Output(f"g-{key}", "style"),
        Input("state", "value"),
        *[Input(c, "value") for c in controls],
    )
    def _update(sel, *values):
        fig, cap = func(B, sel, *values)
        return fig, cap, {"height": f"{int(fig.layout.height or 560)}px"}


for _key, (_func, _controls) in CHARTS.items():
    _register(_key, _func, _controls)


# Clicking a state on a chart highlights it everywhere; clicking it again clears the highlight.
CLICKABLE = ["map", "lollipop", "budget", "index", "ruralurban", "hero", "value", "confounder", "oop", "srb"]


def state_from_point(point):
    """Find the state name in a clicked point (map location, hover text, y label or custom data)."""
    for key in ("location", "text", "y", "customdata"):
        v = point.get(key)
        if isinstance(v, list) and v:
            v = v[0]
        if isinstance(v, str):
            v = v.replace("<b>", "").replace("</b>", "").strip()
            if v in STATES:
                return v
    return None


@app.callback(
    Output("state", "value"),
    *[Input(f"g-{k}", "clickData") for k in CLICKABLE],
    State("state", "value"),
    prevent_initial_call=True,
)
def click_to_highlight(*args):
    *clicks, current = args
    data = clicks[CLICKABLE.index(ctx.triggered_id[2:])] if ctx.triggered_id else None
    if not data or not data.get("points"):
        return no_update
    picked = state_from_point(data["points"][0])
    if picked is None:
        return no_update
    return None if picked == current else picked


if __name__ == "__main__":
    logging.getLogger("werkzeug").setLevel(logging.ERROR)  # show errors only, not every request
    app.run(debug=False, port=8050)