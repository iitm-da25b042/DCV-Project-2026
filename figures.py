"""Chart functions for the dashboard"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import data as D
from theme import (
    AXIS,
    BG,
    CARD,
    FAINT,
    FONT,
    FONT_MONO,
    GRID,
    HL,
    MUTED,
    NEUTRAL,
    REGION_COLOURS,
    REGION_SYMBOLS,
    SEQ,
    SEQ_ALT,
    SEQ_COUNT,
    SERIES_A,
    SERIES_B,
    SUBTLE,
    TEXT,
    base_layout,
    region_legend_traces,
)

inr = D.inr
fmt_p = D.fmt_p
LOG_TICKS = [500, 1000, 2000, 4000, 8000]  # ₹ ticks for log-scale spending axes
LABEL_BG = "rgba(0,0,0,0.75)"


# Helpers
def _log_axis(
    fig,
    title="Health spending per capita (₹, log scale)",
    row=None,
    col=None,
    ticks=LOG_TICKS,
):
    """Log-scale x axis with ₹ tick labels"""
    kw = dict(
        type="log", title_text=title, tickvals=ticks, ticktext=[inr(t) for t in ticks]
    )
    if row:
        fig.update_xaxes(row=row, col=col, **kw)
    else:
        fig.update_xaxes(**kw)


def _lc(label):
    """Converts a label into lower case for use mid-sentence, while retaining acronyms"""
    return " ".join(
        w if (len(w) > 1 and w[:2].isupper()) else w.lower() for w in str(label).split()
    )


def _sig(p):
    """True if p < 0.05 (statistically significant)"""
    return p is not None and not np.isnan(p) and p < 0.05


def _rho_text(s):
    """Spearman results"""
    return f"ρ = {s['r']:+.2f}, {fmt_p(s['p'])}, n = {s['n']}"


def _signed(ind_key, s):
    """Flips the sign of rho so that + always means 'more spending goes with better results'."""
    better = D.IND[ind_key].better if ind_key in D.IND else 1
    out = dict(s)
    out["r"] = s["r"] * better
    lo, hi = s["lo"] * better, s["hi"] * better
    out["lo"], out["hi"] = min(lo, hi), max(lo, hi)
    return out


def _sorted_by_pc(t):
    """Sorts states by spending per capita"""
    return t.sort_values("pc", ascending=True)


def _scatter_regions(
    fig,
    d,
    x,
    y,
    size=None,
    hover=None,
    row=None,
    col=None,
    showlegend=True,
    imputed_col=None,
    sizeref=None,
    base=11,
):
    """One trace for each region: colour and marker shape, with imputed points being drawn as hollow."""
    for r in D.REGION_ORDER:
        mask = (d.Region == r).values
        g = d[mask]
        if g.empty:
            continue
        flags = g[imputed_col] if imputed_col else [False] * len(g)
        sym = [REGION_SYMBOLS[r] + ("-open" if v else "") for v in flags]
        marker = dict(
            color=REGION_COLOURS[r],
            symbol=sym,
            line=dict(color=CARD, width=1.2),
            opacity=0.92,
        )
        if size is not None:
            marker.update(size=g[size], sizemode="area", sizeref=sizeref, sizemin=7)
        else:
            marker.update(size=base)
        tr = go.Scatter(
            x=g[x],
            y=g[y],
            mode="markers",
            name=r,
            legendgroup=r,
            showlegend=showlegend,
            marker=marker,
            customdata=hover[0][mask] if hover else None,
            hovertemplate=hover[1] if hover else None,
            text=g["State"],
        )
        if row:
            fig.add_trace(tr, row=row, col=col)
        else:
            fig.add_trace(tr)


def _highlight_point(
    fig, x, y, label, row=None, col=None, size=15, textpos="top center"
):
    """Rings the selected state in the highlight colour and labels it."""
    tr = go.Scatter(
        x=[x],
        y=[y],
        mode="markers+text",
        text=[f"<b>{label}</b>"],
        textposition=textpos,
        textfont=dict(color=HL, size=12.5),
        showlegend=False,
        hoverinfo="skip",
        marker=dict(size=size, color="rgba(0,0,0,0)", line=dict(color=HL, width=2.5)),
    )
    if row:
        fig.add_trace(tr, row=row, col=col)
    else:
        fig.add_trace(tr)


def _label_points(fig, d, x, y, names, row=None, col=None, pos=None, logx=False):
    """Labels named states with short leader lines and edge-aware positioning."""
    xs = np.log10(d[x]) if logx else d[x]
    xr = (xs.min(), xs.max())
    yr = (d[y].min(), d[y].max())
    for i, nm in enumerate(names):
        g = d[d.State == nm]
        if g.empty:
            continue
        px = float(np.log10(g[x].iloc[0]) if logx else g[x].iloc[0])
        py = float(g[y].iloc[0])
        fx = (px - xr[0]) / (
            (xr[1] - xr[0]) or 1
        )  # position as a fraction of the axis range
        fy = (py - yr[0]) / ((yr[1] - yr[0]) or 1)
        ay = (24 if fy > 0.85 else -24) - (14 if i % 2 else 0) * (
            1 if fy <= 0.85 else -1
        )
        ax = -30 if fx > 0.8 else (30 if fx < 0.12 else [0, -34, 34][i % 3])
        fig.add_annotation(
            x=px,
            y=py,
            text=nm,
            showarrow=True,
            arrowhead=0,
            arrowwidth=1,
            arrowcolor=FAINT,
            ax=ax,
            ay=ay,
            font=dict(size=11.5, color=SUBTLE),
            bgcolor=LABEL_BG,
            xanchor="right" if ax < 0 else ("left" if ax > 0 else "center"),
        )


# Header KPI cards
def kpis(b):
    """Headline numbers shown in cards above the tabs."""
    t = b.total
    lo, hi = t.loc[t.pc.idxmin()], t.loc[t.pc.idxmax()]
    so = D.spearman(t.pc, t.outcome_idx)
    sc = D.spearman(t.pc, t.care_idx)
    si = D.spearman(t.pc, t.imr)
    return [
        dict(
            label="Median health spending per capita",
            value=inr(t.pc.median()),
            note=f"From {inr(lo.pc)} ({lo.State}) to {inr(hi.pc)} ({hi.State})",
        ),
        dict(
            label="Spending vs Outcome Index",
            value=f"ρ = {so['r']:+.2f}",
            accent=True,
            note=f"Higher spenders have healthier mothers and children ({fmt_p(so['p'])})",
        ),
        dict(
            label="Spending vs Care Index",
            value=f"ρ = {sc['r']:+.2f}",
            note=f"No clear link with care received ({fmt_p(sc['p'])})",
        ),
        dict(
            label="Spending vs infant mortality",
            value=f"ρ = {si['r']:+.2f}",
            note=f"Higher spenders lose fewer babies in the first year ({fmt_p(si['p'])})",
        ),
    ]


# Tab 1: Spending Landscape
def map_spending(b, sel):
    """Choropleth of spending per capita (log colour scale). Grey states are not in the study."""
    t = b.total
    names = [f["properties"]["st_nm"] for f in b.geojson["features"]]
    nodata = [n for n in names if n not in set(t.State)]
    fig = go.Figure()
    # Layer 1: States/UTs not in the study, in flat grey
    fig.add_trace(
        go.Choropleth(
            geojson=b.geojson,
            featureidkey="properties.st_nm",
            locations=nodata,
            z=[0] * len(nodata),
            colorscale=[[0, GRID], [1, GRID]],
            showscale=False,
            marker_line_color=AXIS,
            marker_line_width=0.5,
            hovertemplate="%{location}<br>No data in this study<extra></extra>",
        )
    )
    # Layer 2: 29 states, coloured by spending per capita
    ticks = [500, 1000, 2000, 4000]
    fig.add_trace(
        go.Choropleth(
            geojson=b.geojson,
            featureidkey="properties.st_nm",
            locations=t.State,
            z=np.log10(t.pc),
            colorscale=SEQ,
            zmin=np.log10(450),
            zmax=np.log10(6500),
            marker_line_color=CARD,
            marker_line_width=0.7,
            customdata=np.stack(
                [t.pc.map(inr), t.spend_cr.map(inr), t.pop_m.round(1), t.Region], axis=1
            ),
            hovertemplate="<b>%{location}</b> (%{customdata[3]})<br>%{customdata[0]} per capita"
            "<br>Total: %{customdata[1]} crore<br>Population: %{customdata[2]} million<extra></extra>",
            colorbar=dict(
                title=dict(
                    text="₹ per capita", side="top", font=dict(color=MUTED, size=12)
                ),
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.02,
                yanchor="top",
                tickvals=np.log10(ticks),
                ticktext=[inr(v) for v in ticks],
                tickfont=dict(family=FONT_MONO, color=FAINT, size=11),
                thickness=10,
                len=0.45,
                outlinewidth=0,
            ),
        )
    )
    # Layer 3: Outline the selected state in the highlight colour
    if sel in set(t.State):
        fig.add_trace(
            go.Choropleth(
                geojson=b.geojson,
                featureidkey="properties.st_nm",
                locations=[sel],
                z=[1],
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False,
                marker_line_color=HL,
                marker_line_width=3,
                hoverinfo="skip",
            )
        )
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)",
        projection_type="mercator",
    )
    base_layout(
        fig,
        "Health spending per capita by state",
        "Actual state health expenditure (Accounts) ÷ Population · Click a state to highlight it",
        height=680,
        legend=False,
        margin=dict(l=8, r=8, b=70),
    )
    gap = t.pc.max() / t.pc.min()
    cap = (
        f"Per capita expenditure ranges from {inr(t.pc.min())} in {t.loc[t.pc.idxmin(), 'State']} to "
        f"{inr(t.pc.max())} in {t.loc[t.pc.idxmax(), 'State']}, a {gap:.0f}-fold gap. "
        f"The brightest states are mostly small hilly and north-eastern states, while the large "
        f"plains states of the north and east sit at the dark end; the median is {inr(t.pc.median())}."
    )
    return fig, cap


def lollipop_spending(b, sel):
    """Per capita expenditure of every state, ranked, with the median marked."""
    t = _sorted_by_pc(b.total)
    med = t.pc.median()
    fig = go.Figure()
    # Grey stems from 0 to the value of each state
    xs, ys = [], []
    for _, r in t.iterrows():
        xs += [0, r.pc, None]
        ys += [r.State, r.State, None]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=AXIS, width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    # Region-coloured, region-shaped heads
    for reg in D.REGION_ORDER:
        g = t[t.Region == reg]
        fig.add_trace(
            go.Scatter(
                x=g.pc,
                y=g.State,
                mode="markers",
                name=reg,
                marker=dict(
                    color=REGION_COLOURS[reg],
                    symbol=REGION_SYMBOLS[reg],
                    size=13,
                    line=dict(color=CARD, width=1),
                ),
                customdata=np.stack([g.pc.map(inr), g.pop_m.round(1)], axis=1),
                hovertemplate="<b>%{y}</b><br>%{customdata[0]} per capita"
                "<br>%{customdata[1]} million people<extra></extra>",
            )
        )
    # Median acts as the reference line
    fig.add_vline(x=med, line=dict(color=FAINT, width=1.2, dash="dash"))
    fig.add_annotation(
        x=med,
        y=0,
        yref="paper",
        text=f"Median {inr(med)}",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        xshift=6,
        yshift=4,
        font=dict(color=SUBTLE, size=11.5),
    )
    if sel in set(t.State):
        r = t[t.State == sel].iloc[0]
        _highlight_point(fig, r.pc, r.State, f"  {inr(r.pc)}", textpos="middle right")
    lo, hi = t.iloc[0], t.iloc[-1]
    base_layout(
        fig,
        "Health spending per capita, ranked by state",
        f"Coloured and shaped by region · {hi.State} spends {hi.pc / lo.pc:.0f}× more per capita "
        f"than {lo.State}",
        height=760,
    )
    fig.update_xaxes(
        title_text="Health spending per capita (₹)",
        tickprefix="₹",
        tickformat=",",
        rangemode="tozero",
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=list(t.State),
        gridcolor="rgba(0,0,0,0)",
        tickfont=dict(family=FONT, size=11.5, color=SUBTLE),
    )
    top = t.nlargest(8, "pc")
    small = int((top.pop_m < 15).sum())
    cap = (
        f"Half the states spend less than {inr(med)} per capita. "
        f"{'All' if small == 8 else small} of the 8 highest spenders have fewer than 15 million people, and "
        f"{(top.Region == 'Northeast').sum()} of those 8 are in the Northeast, where small populations "
        f"and hard terrain raise the cost of reaching each person."
    )
    return fig, cap


def _ordinal(n):
    """Converts cardinal to ordinal number"""
    n = int(n)
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def rank_shift(b, sel):
    """Slope chart ranks states by total spending (left) vs rank by spending per capita (right)."""
    t = b.total.copy()
    t["r_total"] = t.spend_cr.rank(ascending=False).astype(int)
    t["r_pc"] = t.pc.rank(ascending=False).astype(int)
    t["move"] = t.r_total - t.r_pc
    fig = go.Figure()
    # One line per state joining its total-spending rank to its per-capita rank
    for _, r in t.iterrows():
        is_sel = r.State == sel
        col = HL if is_sel else REGION_COLOURS[r.Region]
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[r.r_total, r.r_pc],
                mode="lines+markers",
                line=dict(color=col, width=3.5 if is_sel else 1.3),
                opacity=1 if is_sel or not sel else 0.5,
                marker=dict(
                    size=9 if is_sel else 7, color=col, symbol=REGION_SYMBOLS[r.Region]
                ),
                name=r.Region,
                legendgroup=r.Region,
                showlegend=False,
                customdata=[[r.State, inr(r.spend_cr), inr(r.pc)]] * 2,
                hovertemplate="<b>%{customdata[0]}</b><br>Total %{customdata[1]} crore"
                "<br>%{customdata[2]} per capita<extra></extra>",
            )
        )
    # State names and ranks on both sides
    for _, r in t.iterrows():
        c = HL if r.State == sel else SUBTLE
        fig.add_annotation(
            x=0,
            y=r.r_total,
            text=f"{r.State}  {r.r_total}",
            xanchor="right",
            xshift=-8,
            showarrow=False,
            font=dict(size=11, color=c),
        )
        fig.add_annotation(
            x=1,
            y=r.r_pc,
            text=f"{r.r_pc}  {r.State}",
            xanchor="left",
            xshift=8,
            showarrow=False,
            font=dict(size=11, color=c),
        )
    region_legend_traces(fig, [r for r in D.REGION_ORDER if r in set(t.Region)])
    base_layout(
        fig,
        "A large budget does not imply high per capita expenditure",
        "Rank by total health spending (left) vs rank by spending per capita (right) · 1 = highest",
        height=640,
        margin=dict(l=150, r=150),
    )
    fig.update_xaxes(visible=False, range=[-0.05, 1.05])
    fig.update_yaxes(autorange="reversed", visible=False)
    up = t.loc[t.move.idxmax()]
    down = t.loc[t.move.idxmin()]
    cap = (
        f"{down.State} has the {_ordinal(down.r_total)}-largest health budget in total but ranks "
        f"{_ordinal(down.r_pc)} per capita, while {up.State} jumps from {_ordinal(up.r_total)} to "
        f"{_ordinal(up.r_pc)}. Totals mostly measure state size, which is why every spending comparison is reported in "
        f"per capita terms."
    )
    return fig, cap


def treemap_spending(b, sel):
    """Region > state treemap: box size = total spending, colour = spending per capita."""
    t = b.total
    reg = (
        t.groupby("Region")
        .agg(spend_cr=("spend_cr", "sum"), pop=("Population (Total)", "sum"))
        .reset_index()
    )
    reg["pc"] = reg.spend_cr * 1e7 / reg["pop"]
    # Three levels: India -> regions -> states
    ids = ["India"] + list(reg.Region) + list(t.Region + "/" + t.State)
    labels = ["India"] + list(reg.Region) + list(t.State)
    parents = [""] + ["India"] * len(reg) + list(t.Region)
    values = [t.spend_cr.sum()] + list(reg.spend_cr) + list(t.spend_cr)
    pcs = (
        [t.spend_cr.sum() * 1e7 / t["Population (Total)"].sum()]
        + list(reg.pc)
        + list(t.pc)
    )
    # Thick highlight-coloured border for selected state
    line_cols = [CARD] * (1 + len(reg)) + [HL if s == sel else CARD for s in t.State]
    line_w = [1] * (1 + len(reg)) + [4 if s == sel else 1 for s in t.State]
    ticks = [500, 1000, 2000, 4000]
    fig = go.Figure(
        go.Treemap(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(
                colors=np.log10(pcs),
                colorscale=SEQ,
                cmin=np.log10(450),
                cmax=np.log10(6500),
                line=dict(color=line_cols, width=line_w),
                colorbar=dict(
                    title=dict(
                        text="₹ per capita", side="top", font=dict(color=MUTED, size=12)
                    ),
                    orientation="h",
                    x=0.5,
                    xanchor="center",
                    y=-0.02,
                    yanchor="top",
                    tickvals=np.log10(ticks),
                    ticktext=[inr(v) for v in ticks],
                    tickfont=dict(family=FONT_MONO, color=FAINT, size=11),
                    thickness=10,
                    len=0.45,
                    outlinewidth=0,
                ),
            ),
            customdata=np.stack(
                [[inr(v) for v in values], [inr(v) for v in pcs]], axis=1
            ),
            texttemplate="<b>%{label}</b><br>%{customdata[1]} per capita",
            hovertemplate="<b>%{label}</b><br>Total %{customdata[0]} crore<br>%{customdata[1]} per capita"
            "<br>%{percentRoot:.1%} of all spending<extra></extra>",
            textfont=dict(family=FONT, size=12),
            root_color="rgba(0,0,0,0)",
            tiling=dict(pad=2),
            pathbar=dict(visible=False),
        )
    )
    fig.update_layout(
        uniformtext=dict(minsize=9, mode="hide")
    )  # Hide labels that do not fit
    base_layout(
        fig,
        "Where is the health money and how thinly is it spread?",
        "Box size = total health spending (₹ crore) · Colour = spending per capita · "
        "Click a region to zoom",
        height=680,
        legend=False,
        margin=dict(l=8, r=8, b=70),
    )
    top5 = t.nlargest(5, "spend_cr")
    share_spend = top5.spend_cr.sum() / t.spend_cr.sum() * 100
    share_pop = top5["Population (Total)"].sum() / t["Population (Total)"].sum() * 100
    ne = reg[reg.Region == "Northeast"].iloc[0]
    cap = (
        f"The five biggest budgets ({', '.join(top5.State)}) make up {share_spend:.0f}% of all state "
        f"health spending but serve {share_pop:.0f}% of the population. The boxes for the Northeastern states are small "
        f"but bright: together this region spends {inr(ne.pc)} per capita, against {inr(pcs[0])} for all 29 "
        f"states combined."
    )
    return fig, cap


def spending_histogram(b, sel):
    """Distribution of spending per capita: histogram, a rug of ticks (one per state), median and mean."""
    t = b.total
    fig = go.Figure()
    # Bars coloured by how many states they hold (taller = brighter), with a colour bar as the key
    edges = np.arange(0, 7000, 500)
    counts, _ = np.histogram(t.pc, bins=edges)
    mids = (edges[:-1] + edges[1:]) / 2
    fig.add_trace(
        go.Bar(
            x=mids,
            y=counts,
            width=480,
            showlegend=False,
            marker=dict(
                color=counts,
                colorscale=SEQ_COUNT,
                cmin=0,
                cmax=int(counts.max()),
                line=dict(width=0),
                colorbar=dict(
                    title=dict(
                        text="Number<br>of states", font=dict(color=MUTED, size=12)
                    ),
                    tickvals=list(range(0, int(counts.max()) + 1, 2)),
                    tickfont=dict(family=FONT_MONO, color=FAINT, size=11),
                    thickness=10,
                    len=0.55,
                    y=0.62,
                    outlinewidth=0,
                ),
            ),
            customdata=[f"{inr(lo)}–{inr(lo + 500)}" for lo in edges[:-1]],
            hovertemplate="%{customdata}<br>%{y} states<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t.pc,
            y=[-0.6] * len(t),
            mode="markers",
            showlegend=False,
            text=t.State,
            marker=dict(symbol="line-ns", size=16, line=dict(color=SUBTLE, width=1.6)),
            customdata=t.pc.map(inr),
            hovertemplate="<b>%{text}</b><br>%{customdata} per capita<extra></extra>",
        )
    )
    med, mean = t.pc.median(), t.pc.mean()
    fig.add_vline(x=med, line=dict(color=FAINT, width=1.2, dash="dash"))
    fig.add_annotation(
        x=med,
        y=1,
        yref="paper",
        text=f"Median {inr(med)}",
        showarrow=False,
        xanchor="left",
        xshift=4,
        yanchor="top",
        font=dict(color=SUBTLE, size=11.5),
    )
    fig.add_vline(x=mean, line=dict(color=FAINT, width=1.2, dash="dot"))
    fig.add_annotation(
        x=mean,
        y=0.9,
        yref="paper",
        text=f"Mean {inr(mean)}",
        showarrow=False,
        xanchor="left",
        xshift=4,
        yanchor="top",
        font=dict(color=FAINT, size=11.5),
    )
    if sel in set(t.State):
        r = t[t.State == sel].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[r.pc],
                y=[-0.6],
                mode="markers+text",
                text=[f"<b>{sel}</b>"],
                textposition="bottom center",
                textfont=dict(color=HL, size=12),
                showlegend=False,
                hoverinfo="skip",
                marker=dict(symbol="line-ns", size=22, line=dict(color=HL, width=3)),
            )
        )
    base_layout(
        fig,
        "Most states cluster at the low end",
        "Number of states in each ₹500 band of spending per capita · ticks below show each state",
        height=640,
        legend=False,
    )
    fig.update_xaxes(
        title_text="Health spending per capita (₹)", tickprefix="₹", tickformat=","
    )
    fig.update_yaxes(title_text="Number of states", range=[-1.6, None], dtick=2)
    n15 = int((t.pc < 1500).sum())
    n4k = int((t.pc > 4000).sum())
    cap = (
        f"{n15} of {len(t)} states spend under ₹1,500 per capita, while a long tail of {n4k} states "
        f"spends over ₹4,000. Since the mean ({inr(mean)}) is pulled well above the median "
        f"({inr(med)}), rank-based correlations (Spearman) and log scales are used."
    )
    return fig, cap


def budget_credibility(b, sel):
    """Actual spending and the Revised Estimate as % of the Budget Estimate for 2019-20."""
    t = _sorted_by_pc(b.total)
    fig = go.Figure()
    # Grey bar from 100% to the actual, so the size of the shortfall/overshoot is visible
    xs, ys = [], []
    for _, r in t.iterrows():
        xs += [min(r.util_be, 100), max(r.util_be, 100), None]
        ys += [r.State, r.State, None]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=AXIS, width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t.util_re,
            y=t.State,
            mode="markers",
            name="Revised Estimate",
            marker=dict(
                symbol="diamond-open", size=11, color=SERIES_A, line=dict(width=2)
            ),
            customdata=t.pc_re.map(inr),
            hovertemplate="<b>%{y}</b><br>Revised Estimate: %{x:.1f}% of budget"
            "<br>(%{customdata} per capita)<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t.util_be,
            y=t.State,
            mode="markers",
            name="Actual spending (Accounts)",
            marker=dict(
                symbol="circle", size=11, color=SERIES_B, line=dict(color=CARD, width=1)
            ),
            customdata=np.stack([t.pc.map(inr), t.pc_be.map(inr)], axis=1),
            hovertemplate="<b>%{y}</b><br>Actual: %{x:.1f}% of budget"
            "<br>%{customdata[0]} spent vs %{customdata[1]} budgeted per capita<extra></extra>",
        )
    )
    fig.add_vline(x=100, line=dict(color=FAINT, width=1.2))
    fig.add_annotation(
        x=100,
        y=0,
        yref="paper",
        text="100% = spent as budgeted",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        xshift=-6,
        yshift=4,
        font=dict(color=SUBTLE, size=11.5),
        bgcolor=LABEL_BG,
    )
    if sel in set(t.State):
        r = t[t.State == sel].iloc[0]
        _highlight_point(
            fig, r.util_be, r.State, f"  {r.util_be:.0f}%", textpos="middle right"
        )
    base_layout(
        fig,
        "Most states spent less on health than they budgeted",
        "2019-20 actual spending and Revised Estimate as % of the Budget Estimate · "
        "states in the same order as the spending ranking",
        height=760,
    )
    fig.update_xaxes(title_text="% of Budget Estimate (2019-20)", ticksuffix="%")
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=list(t.State),
        gridcolor="rgba(0,0,0,0)",
        tickfont=dict(family=FONT, size=11.5, color=SUBTLE),
    )
    under = t[t.util_be < 100]
    lo, hi = t.loc[t.util_be.idxmin()], t.loc[t.util_be.idxmax()]
    cap = (
        f"In 2019-20, {len(under)} of {len(t)} states spent less on health than they budgeted, by a "
        f"median of {100 - under.util_be.median():.0f}%. {lo.State} spent only {lo.util_be:.0f}% of its "
        f"Budget Estimate, while {hi.State} spent {hi.util_be:.0f}%. Budgets therefore "
        f"overstate the actual expenditure on healthcare in several states."
    )
    return fig, cap


# Tab 2: Health Outcomes
# Indicator groups: outcome groups first followed by care groups.
# Each group has its own colour scale (lighter is better)
HEATMAP_GROUPS = {
    "Outcomes": [
        ("Child survival", ["nmr", "imr", "u5mr"], "Viridis"),
        ("Child nutrition", ["stunt", "wast", "underwt", "anaem_child"], "Darkmint_r"),
        ("Mothers' health", ["anaem_preg", "anaem_women", "lowbmi"], "RdPu_r"),
    ],
    "Care": [
        ("Pregnancy & birth", ["anc1", "anc4", "ifa", "inst", "pnc"], "YlOrBr_r"),
        ("Young children", ["vacc", "vita", "ebf", "diet"], "Purples_r"),
    ],
}


def split_heatmap(b, sel, part="Outcomes"):
    """Heatmap of scores ranging from 0-100 for the outcome/care indicators. There is one coloured block per group
    and each block has its own colour bar beneath it. The states are represented as rows, sorted by spending
    """
    t = b.total.sort_values("pc", ascending=False).reset_index(drop=True)
    groups = [
        (g, [k for k in keys if k in t and t[k].notna().any()], sc)
        for g, keys, sc in HEATMAP_GROUPS[part]
    ]
    fig = make_subplots(
        rows=1,
        cols=len(groups),
        shared_yaxes=True,
        horizontal_spacing=0.03,
        column_widths=[len(k) for _, k, _ in groups],
        subplot_titles=[g for g, _, _ in groups],
    )
    spend = t.pc.map(inr).to_numpy(dtype=object)
    all_scores = []
    for ci, (g, keys, scale) in enumerate(groups, start=1):
        # 0-100 score per indicator, flipped where lower is better, so that 100 represents the best state
        z = np.column_stack([D.minmax_score(t[k], D.IND[k].better) for k in keys])
        all_scores.append(z)
        actual = np.column_stack([t[k].round(1).to_numpy(dtype=object) for k in keys])
        units = np.array([[D.IND[k].unit for k in keys]] * len(t), dtype=object)
        sp = np.repeat(spend[:, None], len(keys), axis=1)
        x0, x1 = fig.layout[f"xaxis{'' if ci == 1 else ci}"].domain
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=[D.IND[k].label for k in keys],
                y=t.State,
                zmin=0,
                zmax=100,
                colorscale=scale,
                customdata=np.dstack([actual, units, sp]),
                xgap=2,
                ygap=2,
                hovertemplate="<b>%{y}</b> (%{customdata[2]} per capita)<br>%{x}: %{customdata[0]} %{customdata[1]}"
                "<br>Score %{z:.0f} / 100<extra></extra>",
                # Small horizontal colourbar under each block
                colorbar=dict(
                    orientation="h",
                    x=(x0 + x1) / 2,
                    xanchor="center",
                    y=-0.25,
                    yanchor="top",
                    len=(x1 - x0) * 0.9,
                    thickness=8,
                    outlinewidth=0,
                    tickvals=[0, 100],
                    ticktext=["worst", "best"],
                    tickfont=dict(family=FONT_MONO, color=FAINT, size=10),
                ),
            ),
            row=1,
            col=ci,
        )
    if sel in set(t.State):
        i = int(t.index[t.State == sel][0])
        for ci in range(1, len(groups) + 1):
            for colour, width in ((BG, 6), (HL, 2.5)):
                fig.add_shape(
                    type="rect",
                    xref=f"x{'' if ci == 1 else ci} domain",
                    yref="y",
                    x0=0,
                    x1=1,
                    y0=i - 0.5,
                    y1=i + 0.5,
                    line=dict(color=colour, width=width),
                )
    mean = np.hstack(all_scores).mean(axis=1)
    s = D.spearman(t.pc, mean)
    title = (
        "Health outcomes: States at the top with higher expenditure show better performance"
        if part == "Outcomes"
        else "Care received: No top-to-bottom pattern observed"
    )
    base_layout(
        fig,
        title,
        "States sorted in descending order by per capita expenditure · lighter = better in all the blocks",
        height=860,
        legend=False,
        margin=dict(l=140, b=190),
    )
    fig.update_xaxes(
        tickangle=-40,
        gridcolor="rgba(0,0,0,0)",
        tickfont=dict(family=FONT, size=11, color=SUBTLE),
    )
    fig.update_yaxes(
        autorange="reversed",
        gridcolor="rgba(0,0,0,0)",
        tickfont=dict(family=FONT, size=11.5, color=SUBTLE),
    )
    third = len(t) // 3
    top, bot = mean[:third].mean(), mean[-third:].mean()
    n_ind = sum(len(k) for _, k, _ in groups)
    what = "health outcomes" if part == "Outcomes" else "care indicators"
    r = s["r"]
    stats_txt = f"Spearman's ρ = {r:.2f}, {fmt_p(s['p'])}, n = {s['n']}"
    if _sig(s["p"]):
        strength = (
            "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
        )
        direction = "positive" if r > 0 else "negative"
        second = (
            f"This difference is backed by a {strength} {direction} correlation between "
            f"spending and {what} ({stats_txt})."
        )
    else:
        second = f"There is no significant correlation between spending and {what} ({stats_txt})."
    cap = (
        f"Across the {n_ind} {what}, the highest-spending third of states averaged {top:.0f}/100, "
        f"compared to {bot:.0f}/100 for the lowest-spending third. {second}"
    )
    return fig, cap


def index_scatter(b, sel):
    """Outcome Index vs Care Index, split into quadrants at the medians."""
    t = b.total
    mo, mc = t.outcome_idx.median(), t.care_idx.median()
    fig = go.Figure()
    # Quadrant guides at the medians
    fig.add_vline(x=mc, line=dict(color=FAINT, width=1, dash="dot"))
    fig.add_hline(y=mo, line=dict(color=FAINT, width=1, dash="dot"))
    for x, y, text, xa, ya in [
        (1, 1, "More care, better outcomes", "right", "top"),
        (0, 1, "Less care, better outcomes", "left", "top"),
        (0, 0, "Less care, worse outcomes", "left", "bottom"),
        (1, 0, "More care, worse outcomes", "right", "bottom"),
    ]:
        fig.add_annotation(
            x=x,
            y=y,
            xref="x domain",
            yref="y domain",
            text=text,
            showarrow=False,
            xanchor=xa,
            yanchor=ya,
            font=dict(color=FAINT, size=11.5),
        )
    hover = (
        np.stack(
            [t.State, t.pc.map(inr), t.outcome_idx.round(0), t.care_idx.round(0)],
            axis=1,
        ),
        "<b>%{customdata[0]}</b> (%{customdata[1]} per capita)<br>Outcome Index %{customdata[2]}"
        "<br>Care Index %{customdata[3]}<extra></extra>",
    )
    _scatter_regions(fig, t, "care_idx", "outcome_idx", hover=hover, base=13)
    # States where care and outcomes disagree the most
    gap = t.outcome_idx - t.care_idx
    lab = t.loc[list(gap.nlargest(2).index) + list(gap.nsmallest(2).index)]
    lab = lab[lab.State != sel]
    fig.add_trace(
        go.Scatter(
            x=lab.care_idx,
            y=lab.outcome_idx,
            mode="text",
            showlegend=False,
            hoverinfo="skip",
            text=[
                f"{n}   " if x > 60 else f"   {n}"
                for n, x in zip(lab.State, lab.care_idx)
            ],
            textposition=[
                "middle left" if x > 60 else "middle right" for x in lab.care_idx
            ],
            textfont=dict(size=11.5, color=SUBTLE),
        )
    )
    if sel in set(t.State):
        r = t[t.State == sel].iloc[0]
        _highlight_point(fig, r.care_idx, r.outcome_idx, sel, size=18)
    base_layout(
        fig,
        "More care does not always result in good outcomes",
        "Each dot is a state · dotted lines are the medians · both indices 0 (worst state) to 100 (best state)",
        height=760,
    )
    fig.update_xaxes(
        title_text="Care Index (0–100, 6 care indicators)",
        range=[-5, 105],
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Outcome Index (0–100, 7 outcome indicators)",
        range=[-5, 105],
        zeroline=False,
    )
    s = D.spearman(t.care_idx, t.outcome_idx)
    q = t[(t.care_idx < mc) & (t.outcome_idx >= mo)]
    ne = int((q.Region == "Northeast").sum())
    cap = (
        f"Care and outcomes are only {'weakly' if abs(s['r']) < 0.4 else 'moderately'} related across states "
        f"({_rho_text(s)}). {len(q)} states get better-than-median outcomes with less-than-median care, with "
        f"{ne} of them being in the Northeast, which suggests that other factors extending beyond the health system, such as diet, "
        f"sanitation or women's schooling, also shape outcomes."
    )
    return fig, cap


def rural_urban(b, sel, ind="imr"):
    """Dumbbell: urban vs rural value of one indicator for every state, rows sorted by spending."""
    c = b.clean
    u = c[c.Area == "Urban"].set_index("State")
    r = c[c.Area == "Rural"].set_index("State")
    imp = b.imputed.set_index(["State", "Area"])[ind]
    t = b.total.set_index("State").sort_values("pc")
    states = list(t.index)
    lab, unit = D.IND[ind].label, D.IND[ind].unit
    fig = go.Figure()
    for st in states:
        is_sel = st == sel
        fig.add_trace(
            go.Scatter(
                x=[u.loc[st, ind], r.loc[st, ind]],
                y=[st, st],
                mode="lines",
                line=dict(color=HL if is_sel else AXIS, width=4 if is_sel else 2.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    for area, frame, col, sym in [
        ("Urban", u, SERIES_A, "circle"),
        ("Rural", r, SERIES_B, "diamond"),
    ]:
        flags = [bool(imp.get((st, area), False)) for st in states]
        fig.add_trace(
            go.Scatter(
                x=[frame.loc[st, ind] for st in states],
                y=states,
                mode="markers",
                name=area,
                marker=dict(
                    color=col,
                    size=11,
                    symbol=[sym + ("-open" if f else "") for f in flags],
                    line=dict(color=col, width=1.5),
                ),
                customdata=np.array(
                    [
                        [inr(t.loc[st, "pc"]), " (imputed)" if f else ""]
                        for st, f in zip(states, flags)
                    ]
                ),
                hovertemplate="<b>%{y}</b> · %{customdata[0]} per capita<br>"
                + area
                + ": %{x:.1f}%{customdata[1]}<extra></extra>",
            )
        )
    if sel in states:
        fig.add_trace(
            go.Scatter(
                x=[u.loc[sel, ind], r.loc[sel, ind]],
                y=[sel, sel],
                mode="markers",
                marker=dict(
                    size=19, color="rgba(0,0,0,0)", line=dict(color=HL, width=2.5)
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    base_layout(
        fig,
        f"Rural vs urban: {_lc(lab)}",
        f"{unit} · States sorted in descending order by per capita spending · hollow markers = imputed values",
        height=760,
    )
    fig.update_xaxes(title_text=f"{lab} ({unit})")
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=states,
        gridcolor="rgba(0,0,0,0)",
        tickfont=dict(family=FONT, size=11.5, color=SUBTLE),
    )
    gap = (r[ind] - u[ind]).reindex(states)  # Rural - Urban
    worse_gap = gap * D.IND[ind].better  # If negative, rural is worse
    s = D.spearman(t.pc.reindex(states), worse_gap)
    worse = int((worse_gap < 0).sum())
    gunit = "percentage points" if unit.startswith("%") else unit
    if _sig(s["p"]):
        tail = f"The gap is {'smaller' if s['r'] > 0 else 'larger'} in higher-spending states ({_rho_text(s)})."
    else:
        tail = (
            f"Increased spending does not clearly narrow the rural-urban gap ({_rho_text(s)}). The extra money is not "
            f"obviously reaching mothers and children in rural areas first."
        )
        better_rural = [st for st in states if worse_gap[st] > 0]          # states where rural does better
    if 1 <= len(better_rural) <= 3:
        names = " and ".join([", ".join(better_rural[:-1]), better_rural[-1]] if len(better_rural) > 1
                             else better_rural)
        exc = f" The {'exception is' if len(better_rural) == 1 else 'exceptions are'} {names}."
    else:
        exc = ""
    cap = (f"Rural areas perform worse than urban ones on {_lc(lab)} in {worse} of {len(states)} states, with a "
           f"median gap of {abs(gap.median()):.1f} {gunit}.{exc} " + tail)
    return fig, cap


HEADLINE = ["imr", "stunt", "anaem_child", "anaem_preg"]


def outcome_spread(b, sel):
    """Box plot with one dot per state for four headline outcomes to show how far apart the states are."""
    t = b.total.reset_index(drop=True)
    rng = np.random.default_rng(7)
    jitter = rng.uniform(-0.28, 0.28, len(t))
    fig = make_subplots(
        rows=1,
        cols=len(HEADLINE),
        horizontal_spacing=0.08,
        subplot_titles=[D.IND[k].label for k in HEADLINE],
    )
    for i, k in enumerate(HEADLINE, start=1):
        fig.add_trace(
            go.Box(
                y=t[k],
                x=[0] * len(t),
                boxpoints=False,
                width=0.8,
                showlegend=False,
                fillcolor="rgba(255,255,255,0.04)",
                line=dict(color=FAINT, width=1.3),
                hoverinfo="skip",
            ),
            row=1,
            col=i,
        )
        for reg in D.REGION_ORDER:
            m = (t.Region == reg).values
            if not m.any():
                continue
            fig.add_trace(
                go.Scatter(
                    x=jitter[m],
                    y=t.loc[m, k],
                    mode="markers",
                    name=reg,
                    legendgroup=reg,
                    showlegend=i == 1,
                    marker=dict(
                        color=REGION_COLOURS[reg],
                        symbol=REGION_SYMBOLS[reg],
                        size=10,
                        line=dict(color=CARD, width=1),
                    ),
                    text=t.loc[m, "State"],
                    customdata=t.loc[m, "pc"].map(inr),
                    hovertemplate="<b>%{text}</b> (%{customdata} per capita)<br>"
                    + D.IND[k].label
                    + ": %{y}<extra></extra>",
                ),
                row=1,
                col=i,
            )
        if sel in set(t.State):
            j = int(t.index[t.State == sel][0])
            _highlight_point(
                fig,
                jitter[j],
                t.loc[j, k],
                sel if i == 1 else "",
                row=1,
                col=i,
                size=16,
                textpos="middle right",
            )
        fig.update_xaxes(visible=False, range=[-0.6, 0.6], row=1, col=i)
        fig.update_yaxes(title_text=D.IND[k].unit, row=1, col=i)
    base_layout(
        fig,
        "How far apart the states are on four headline outcomes",
        "Each dot is a state · the box covers the middle half of states with the median line · "
        "lower is better for all four outcomes",
        height=760,
    )
    i1, i2 = t.loc[t.imr.idxmin()], t.loc[t.imr.idxmax()]
    s1, s2 = t.loc[t.stunt.idxmin()], t.loc[t.stunt.idxmax()]
    cap = (
        f"Infant mortality ranges from {i1.imr:g} deaths per 1,000 live births in {i1.State} to {i2.imr:g} in "
        f"{i2.State}, over {int(i2.imr / i1.imr)} times higher. Stunting varies from {s1.stunt:g}% "
        f"({s1.State}) to {s2.stunt:g}% ({s2.State}). Anaemia, on the other hand, is prevalent everywhere: "
           f"in half of all states at least {t.anaem_child.median():.0f}% of young children and "
           f"{t.anaem_preg.median():.0f}% of pregnant women are anaemic.")
    return fig, cap


SRB_NATURAL = (
    935,
    971,
)  # 103-107 boys per 100 girls which is the range expected without sex selection
SRB_EXPECTED = 952  # About 105 boys per 100 girls
# The darker the colour, the lower is the sex ratio
SRB_SCALE = [[0, "#c0152f"], [0.35, "#e8604c"], [0.7, "#f4a582"], [1, "#fff5f0"]]
SRB_FLOOR = 880


def sex_ratio_birth(b, sel):
    """Sex ratio at birth by state against the natural range; redder dots = bigger shortfall of girls."""
    t = b.total
    key = next(c for c in t.columns if "sex_ratio_at_birth" in c)
    t = t.sort_values(key).reset_index(drop=True)
    fig = go.Figure()
    fig.add_vrect(
        x0=SRB_NATURAL[0],
        x1=SRB_NATURAL[1],
        fillcolor=FAINT,
        opacity=0.12,
        line_width=0,
    )
    fig.add_vline(x=SRB_EXPECTED, line=dict(color=FAINT, width=1.2, dash="dash"))
    fig.add_annotation(
        x=SRB_EXPECTED,
        y=1,
        yref="paper",
        yanchor="bottom",
        showarrow=False,
        text=f"Natural range ({SRB_NATURAL[0]}–{SRB_NATURAL[1]})",
        font=dict(color=SUBTLE, size=11.5),
    )
    xs, ys = [], []
    for _, r in t.iterrows():
        xs += [SRB_EXPECTED, r[key], None]
        ys += [r.State, r.State, None]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=AXIS, width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t[key],
            y=t.State,
            mode="markers",
            showlegend=False,
            # colour runs from deep red (880 or fewer girls) to white (952+, no shortfall)
            marker=dict(
                size=13,
                color=t[key].clip(SRB_FLOOR, SRB_EXPECTED),
                colorscale=SRB_SCALE,
                cmin=SRB_FLOOR,
                cmax=SRB_EXPECTED,
                line=dict(color=CARD, width=1),
                colorbar=dict(
                    title=dict(
                        text="Girls per<br>1,000 boys",
                        font=dict(color=MUTED, size=11.5),
                    ),
                    tickvals=[SRB_FLOOR, 900, 925, SRB_EXPECTED],
                    ticktext=[f"≤{SRB_FLOOR}", "900", "925", f"{SRB_EXPECTED}+"],
                    tickfont=dict(family=FONT_MONO, color=FAINT, size=11),
                    thickness=10,
                    len=0.5,
                    outlinewidth=0,
                ),
            ),
            customdata=t.pc.map(inr),
            hovertemplate="<b>%{y}</b><br>%{x} girls per 1,000 boys<br>%{customdata} per capita<extra></extra>",
        )
    )
    if sel in set(t.State):
        r = t[t.State == sel].iloc[0]
        _highlight_point(
            fig, r[key], r.State, f"  {r[key]:.0f}", textpos="middle right"
        )
    base_layout(
        fig,
        "Spending does not fix all issues: Many states have too few girls born",
        "Sex ratio at birth, children born in the 5 years before NFHS-5 · darker red = fewer girls",
        height=760,
        legend=False,
    )
    fig.update_xaxes(title_text="Girls born per 1,000 boys")
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=list(t.State),
        gridcolor="rgba(0,0,0,0)",
        tickfont=dict(family=FONT, size=11.5, color=SUBTLE),
    )
    below = t[t[key] < SRB_NATURAL[0]]
    lo = t.iloc[0]
    s_pc = D.spearman(t.pc, t[key])
    s_sc = D.spearman(t.school, t[key])
    # Well-educated states (schooling above median) among those with the fewest girls
    edu = t.head(5)
    edu = list(edu[edu.school > t.school.median()].State[:2])
    if not _sig(s_pc["p"]) and not _sig(s_sc["p"]):
        why = (
            f"This shortfall is not directly explained by health spending ({_rho_text(s_pc)}) or women's schooling "
            f"({_rho_text(s_sc)})"
        )
        why += (
            f". Some of the best-educated states, such as {' and '.join(edu)}, have the lowest ratios, "
            f"which points to a preference for the male child and access to sex-determination technology."
            if edu
            else "."
        )
    else:
        why = (
            f"Health spending ({_rho_text(s_pc)}) and women's schooling ({_rho_text(s_sc)}) are both "
            f"examined; correlation alone does not show either one causes the shortfall."
        )
    cap = (
        f"{len(below)} of {len(t)} states record fewer girls than the natural range allows, lowest in "
        f"{lo.State} ({lo[key]:.0f} per 1,000 boys), a pattern consistent with sex-selective abortion "
        f"(foeticide). {why} Survey estimates for small states are imprecise."
    )
    return fig, cap

# Tab 3: Spending vs Outcomes
def label_any(k):
    """Chart label for an indicator key, including the two indices."""
    return {"outcome_idx": "Outcome Index", "care_idx": "Care Index"}.get(k, D.label_of(k))

def unit_any(k):
    """Unit for an indicator key, including the two indices."""
    return {"outcome_idx": "0–100", "care_idx": "0–100"}.get(k, D.IND[k].unit if k in D.IND else "")

def better_any(k):
    """+1 if higher is better, -1 if lower is better (indices: higher is better)."""
    return 1 if k in ("outcome_idx", "care_idx") else D.IND[k].better

def hero_scatter(b, sel, ind="outcome_idx", area="Total", drop_flagged=False):
    """Spending per capita (log scale) vs any indicator: region-coloured bubbles sized by population,
    a trend line, and options for Urban/Rural figures or leaving out the six flagged NE states."""
    d = b.clean[b.clean.Area == area].copy()
    imp = b.imputed[b.imputed.Area == area].set_index("State")
    d["imp"] = d.State.map(imp[ind]).fillna(False).astype(bool) if ind in imp.columns else False
    if drop_flagged:
        d = d[~d.flagged_ne]
    d = d.dropna(subset=[ind])
    s = D.spearman(d.pc, d[ind])
    fig = go.Figure()
    hov = (np.stack([d.State, d.pc.map(inr), d[ind].round(1), d.pop_m.round(1),
                     np.where(d.imp, " (imputed)", "")], axis=1),
           "<b>%{customdata[0]}</b><br>%{customdata[1]} per capita<br>" + label_any(ind)
           + ": %{customdata[2]}%{customdata[4]}<br>%{customdata[3]} million people<extra></extra>")
    sizeref = 2.0 * d.pop_m.max() / (46 ** 2)       # biggest state ~46 px across
    _scatter_regions(fig, d, "pc", ind, size="pop_m", hover=hov, imputed_col="imp", sizeref=sizeref)
    # Trend line: least squares of the indicator on log10(spending)
    sl, ic = D.log_fit(d.pc, d[ind])
    xs = np.logspace(np.log10(d.pc.min() * 0.92), np.log10(d.pc.max() * 1.08), 50)
    fig.add_trace(go.Scatter(x=xs, y=ic + sl * np.log10(xs), mode="lines", name="Trend",
                             line=dict(color=SUBTLE, width=2, dash="dash"), hoverinfo="skip"))
    # Label the extremes: best and worst on the indicator, highest and lowest spender
    lab = (list(d.nsmallest(1, ind).State) + list(d.nlargest(1, ind).State)
           + list(d.nlargest(1, "pc").State) + list(d.nsmallest(1, "pc").State))
    for nm in [x for x in dict.fromkeys(lab) if x != sel]:
        r = d[d.State == nm].iloc[0]
        rad = np.sqrt(r.pop_m / sizeref) / 2
        fig.add_annotation(x=np.log10(r.pc), y=r[ind], text=nm, showarrow=True, arrowhead=0, arrowwidth=1,
                           arrowcolor=FAINT, ax=0, ay=-(rad + 18), font=dict(size=11.5, color=SUBTLE),
                           bgcolor=LABEL_BG)
    if sel in set(d.State):
        r = d[d.State == sel].iloc[0]
        rad = np.sqrt(r.pop_m / sizeref)
        fig.add_trace(go.Scatter(x=[r.pc], y=[r[ind]], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=rad + 10, color="rgba(0,0,0,0)", line=dict(color=HL, width=2.5))))
        fig.add_annotation(x=np.log10(r.pc), y=r[ind], text=f"<b>{sel}</b>", showarrow=True, arrowhead=0,
                           arrowwidth=1, arrowcolor=HL, ax=0, ay=-(rad / 2 + 22), font=dict(size=12, color=HL),
                           bgcolor=LABEL_BG)
    if d.imp.any():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Hollow = imputed",
                                 marker=dict(symbol="circle-open", size=11, color=SUBTLE)))
    direction = "lower is better" if better_any(ind) == -1 else "higher is better"
    sub = f"{area} figures · {direction} · bubble area = population"
    if drop_flagged:
        sub += " · 6 flagged north-eastern states left out"
    name = label_any(ind) if ind in ("outcome_idx", "care_idx") else _lc(label_any(ind))
    base_layout(fig, f"Health spending per capita vs {name}", sub, height=760)
    _log_axis(fig)
    fig.update_yaxes(title_text=f"{label_any(ind)} ({unit_any(ind)})")
    fig.add_annotation(x=0.99, y=0.02, xref="paper", yref="paper", xanchor="right", yanchor="bottom", showarrow=False,
                       text=f"Spearman {_rho_text(s)}", font=dict(family=FONT_MONO, size=12, color=TEXT),
                       bgcolor=LABEL_BG, bordercolor=AXIS, borderpad=6)
    unit = unit_any(ind)
    def val(v):
        """Value with unit"""
        if unit == "0–100":
            return f"{v:.1f} points"
        return f"{v:.1f}{unit}" if unit.startswith("%") else f"{v:.1f} {unit}"
    what = f"{name} scores" if ind in ("outcome_idx", "care_idx") else name
    strength = "strong" if abs(s["r"]) >= 0.7 else "moderate" if abs(s["r"]) >= 0.4 else "weak"
    good = s["r"] * better_any(ind) > 0
    if _sig(s["p"]):
        first = (f"States that spend more per capita have {'higher' if s['r'] > 0 else 'lower'} "
                 f"({'better' if good else 'worse'}) {what}: a {strength} "
                 f"{'positive' if s['r'] > 0 else 'negative'} correlation ({_rho_text(s)}).")
    else:
        first = f"Spending per capita is not significantly related to {name} ({_rho_text(s)})."
    # Median value in the lowest- and highest-spending quarters, with their spending ranges
    q = pd.qcut(d.pc, 4, labels=False, duplicates="drop")
    lo_q, hi_q = d[q == q.min()], d[q == q.max()]
    second = (f"The median is {val(lo_q[ind].median())} in the lowest-spending quarter of states "
              f"({inr(lo_q.pc.min())}–{inr(lo_q.pc.max())} per capita) and {val(hi_q[ind].median())} in the "
              f"highest-spending quarter ({inr(hi_q.pc.min())}–{inr(hi_q.pc.max())}).")
    # Best and worst states, with what they spend
    best = d.loc[d[ind].idxmin() if better_any(ind) == -1 else d[ind].idxmax()]
    worst = d.loc[d[ind].idxmax() if better_any(ind) == -1 else d[ind].idxmin()]
    third = (f"{best.State} does best ({best[ind]:.1f}) on {inr(best.pc)} per capita, while {worst.State} does "
             f"worst ({worst[ind]:.1f}) on {inr(worst.pc)}.")
    cap = f"{first} {second} {third}"
    # Impact of northeastern states on results
    if not drop_flagged:
        rest = d[~d.flagged_ne]
        s2 = D.spearman(rest.pc, rest[ind])
        if _sig(s["p"]) and _sig(s2["p"]) and np.sign(s2["r"]) == np.sign(s["r"]):
            verdict = "so the pattern holds"
        elif _sig(s["p"]):
            verdict = "the pattern weakens below significance"
        elif _sig(s2["p"]):
            verdict = "a significant link appears"
        else:
            verdict = "there is still no clear link"
        cap += (f" Without the six small north-eastern states, ρ = {s2['r']:+.2f} ({fmt_p(s2['p'])}): "
                f"{verdict}.")
    return fig, cap

def correlation_barcode(b, sel):
    """Spearman correlation of spending with every indicator, with 95% confidence intervals."""
    t = b.total
    groups = [("Outcome", "Health outcomes"), ("Care", "Care received"), ("Cost & cover", "Cost and cover for families")]
    rows = []
    for g, _ in groups:
        for ind in [i for i in D.INDICATORS if i.group == g and t[i.key].notna().sum() >= 10]:
            # Flip sign where lower is better
            s = _signed(ind.key, D.spearman(t.pc, t[ind.key]))
            rows.append(dict(group=g, label=ind.label, **s))
    r = pd.DataFrame(rows)
    counts = [int((r.group == g).sum()) for g, _ in groups]
    fig = make_subplots(rows=len(groups), cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=counts,
                        subplot_titles=[name for _, name in groups])
    shown = set()
    for i, (g, _) in enumerate(groups, start=1):
        gg = r[r.group == g].sort_values("r")
        for sig in (True, False):
            h = gg[gg.p.lt(0.05) == sig]
            if h.empty:
                continue
            col = TEXT if sig else NEUTRAL
            fig.add_trace(go.Scatter(
                x=h.r, y=h.label, mode="markers", name="Significant (p < 0.05)" if sig else "Not significant",
                legendgroup=str(sig), showlegend=sig not in shown,
                marker=dict(size=12, color=col if sig else "rgba(0,0,0,0)", symbol="circle",
                            line=dict(width=2, color=col)),
                error_x=dict(type="data", symmetric=False, array=h.hi - h.r, arrayminus=h.r - h.lo,
                             color=col, thickness=1.4, width=0),
                customdata=np.stack([h.p.map(fmt_p), h.n, h.lo.round(2), h.hi.round(2)], axis=1),
                hovertemplate="<b>%{y}</b><br>ρ = %{x:+.2f} (95% CI %{customdata[2]} to %{customdata[3]})"
                              "<br>%{customdata[0]}, n = %{customdata[1]}<extra></extra>"),
                row=i, col=1)
            shown.add(sig)
        fig.add_vline(x=0, line=dict(color=AXIS, width=1.5), row=i, col=1)
        fig.update_yaxes(categoryorder="array", categoryarray=list(gg.label), row=i, col=1)
    base_layout(fig, "Spending tracks health outcomes, not the care received",
                "Spearman correlation of spending per capita with each indicator · signed so that + means higher "
                "spending accompanies better results · bars = 95% confidence interval",
                height=820, margin=dict(l=200))
    fig.update_xaxes(range=[-1, 1], zeroline=False, dtick=0.25, tickformat="+.2f")
    fig.update_yaxes(tickfont=dict(family=FONT, size=11.5, color=SUBTLE), gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title_text="Spearman ρ (+ = better results with more spending)", row=len(groups), col=1)
    out, care = r[r.group == "Outcome"], r[r.group == "Care"]
    top = out.loc[out.r.idxmax()]
    oop = r[r.label.str.startswith("Out-of-pocket")].iloc[0]
    care_sig = list(care[care.p < 0.05].label)
    n_out = int((out.p < 0.05).sum())
    first = (f"All {len(out)} health outcomes" if n_out == len(out) else f"{n_out} of the {len(out)} health outcomes")
    if care_sig:
        verb = "rises" if len(care_sig) == 1 else "rise"
        care_txt = (f"only {len(care_sig)} of the {len(care)} care indicators {verb} with spending "
                    f"({', '.join(_lc(x) for x in care_sig)})")
    else:
        care_txt = f"none of the {len(care)} care indicators rises with spending"
    oop_txt = (" One result moves the opposite way: families pay more out of their own pocket for a delivery in "
               "higher-spending states." if oop.r < 0 and oop.p < 0.05 else "")
    cap = (f"{first} are clearly better in states that spend more, with the strongest link for "
           f"{_lc(top.label)}. But {care_txt}.{oop_txt} A dot whose bar crosses the middle line has no clear "
           f"link either way.")
    return fig, cap

def quarter_bars(b, sel):
    """Horizontal bars: average score of the lowest- vs highest-spending quarter of states, per indicator group."""
    t = b.total.copy()
    t["q"] = pd.qcut(t.pc, 4, labels=False)
    groups = HEATMAP_GROUPS["Outcomes"] + HEATMAP_GROUPS["Care"]
    names, low, high, mine = [], [], [], []
    for g, keys, _ in groups:
        keys = [k for k in keys if k in t and t[k].notna().any()]
        score = np.column_stack([D.minmax_score(t[k], D.IND[k].better) for k in keys]).mean(axis=1)
        t[g] = score
        names.append(g)
        low.append(t.loc[t.q == 0, g].mean())
        high.append(t.loc[t.q == 3, g].mean())
        mine.append(t.loc[t.State == sel, g].iloc[0] if sel in set(t.State) else None)
    lo_rng = f"{inr(t.loc[t.q == 0, 'pc'].min())}–{inr(t.loc[t.q == 0, 'pc'].max())}"
    hi_rng = f"{inr(t.loc[t.q == 3, 'pc'].min())}–{inr(t.loc[t.q == 3, 'pc'].max())}"
    order = names[::-1]                      # first group at the top
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=low, orientation="h", name=f"Lowest-spending quarter ({lo_rng})",
                         marker=dict(color=SERIES_B, cornerradius=3), text=[f"{v:.0f}" for v in low],
                         textposition="outside", textfont=dict(color=SUBTLE, size=11),
                         hovertemplate="<b>%{y}</b><br>Lowest-spending quarter: %{x:.0f}/100<extra></extra>"))
    fig.add_trace(go.Bar(y=names, x=high, orientation="h", name=f"Highest-spending quarter ({hi_rng})",
                         marker=dict(color=SERIES_A, cornerradius=3), text=[f"{v:.0f}" for v in high],
                         textposition="outside", textfont=dict(color=SUBTLE, size=11),
                         hovertemplate="<b>%{y}</b><br>Highest-spending quarter: %{x:.0f}/100<extra></extra>"))
    if sel in set(t.State):                  # the selected state's own score as a cyan tick
        fig.add_trace(go.Scatter(y=names, x=mine, mode="markers", name=sel,
                                 marker=dict(symbol="line-ns", size=26, line=dict(color=HL, width=3)),
                                 hovertemplate=f"<b>%{{y}}</b><br>{sel}: %{{x:.0f}}/100<extra></extra>"))
    # Divider between the outcome groups and the care groups
    n_out = len(HEATMAP_GROUPS["Outcomes"])
    div = len(names) - n_out - 0.5          # categories are counted from the bottom
    fig.add_hline(y=div, line=dict(color=AXIS, width=1, dash="dot"))
    fig.add_annotation(x=1, xref="paper", y=div, yanchor="bottom", xanchor="right", showarrow=False,
                       text="▲ health outcomes · care received ▼", font=dict(color=FAINT, size=11))
    base_layout(fig, "The top spenders lead on every outcome group, but not on care",
                "Average score of the lowest- and highest-spending quarters · 0 = worst state, 100 = best",
                height=640, margin=dict(l=180))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    fig.update_xaxes(title_text="Average score (0–100)", range=[0, 105])
    fig.update_yaxes(categoryorder="array", categoryarray=order, gridcolor="rgba(0,0,0,0)",
                     tickfont=dict(family=FONT, size=12, color=SUBTLE))
    gaps = {n: h - l for n, l, h in zip(names, low, high)}
    big = max(names[:n_out], key=lambda n: gaps[n])
    care_name = {"Pregnancy & birth": "care in pregnancy and birth", "Young children": "care for young children"}

    def compare(n):
        """'lower on care in pregnancy and birth (43 vs 56)' etc."""
        i = names.index(n)
        word = "about the same" if abs(gaps[n]) < 5 else ("higher" if gaps[n] > 0 else "lower")
        return f"{word} on {care_name.get(n, n.lower())} ({high[i]:.0f} vs {low[i]:.0f})"

    care_txt = " and ".join(compare(n) for n in names[n_out:])
    cap = (f"Mothers and children are far healthier in the quarter of states that spend the most than in the "
           f"quarter that spends the least: on {big.lower()} they score {high[names.index(big)]:.0f} out of 100, "
           f"against {low[names.index(big)]:.0f}. But they do not receive more care: the big spenders score "
           f"{care_txt}.")
    return fig, cap

def value_for_money(b, sel):
    """Diverging bars: each state's Outcome Index minus what its spending predicts (from the trend line)."""
    t = b.total.copy()
    sl, ic = D.log_fit(t.pc, t.outcome_idx)                 # same trend line as the main scatter
    t["expected"] = ic + sl * np.log10(t.pc)
    t["gap"] = t.outcome_idx - t.expected
    t = t.sort_values("gap").reset_index(drop=True)          # plotted bottom to top: best ends on top
    better = t.gap >= 0
    fig = go.Figure()
    for mask, name, col in [(better, "Better than its spending predicts", SERIES_A),
                            (~better, "Worse than its spending predicts", SERIES_B)]:
        g = t[mask]
        fig.add_trace(go.Bar(
            y=g.State, x=g.gap, orientation="h", name=name,
            marker=dict(color=col, cornerradius=3,
                        line=dict(color=[HL if st == sel else col for st in g.State],
                                  width=[3 if st == sel else 0 for st in g.State])),
            text=[f"{v:+.0f}" for v in g.gap], textposition="outside", textfont=dict(color=SUBTLE, size=10.5),
            customdata=np.stack([g.pc.map(inr), g.outcome_idx.round(0), g.expected.round(0)], axis=1),
            hovertemplate="<b>%{y}</b> (%{customdata[0]} per capita)<br>Outcome Index %{customdata[1]}, "
                          "expected %{customdata[2]}<br>%{x:+.0f} points<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1.5))
    base_layout(fig, "Health per rupee: who does better or worse than their spending predicts",
                "Actual Outcome Index minus the value expected from the state's spending (trend line) · points",
                height=760, margin=dict(l=150))
    fig.update_layout(barmode="relative")
    lim = float(np.ceil(t.gap.abs().max() / 10) * 10 + 8)
    fig.update_xaxes(title_text="Points above (+) or below (−) expected Outcome Index", range=[-lim, lim],
                     zeroline=False)
    fig.update_yaxes(categoryorder="array", categoryarray=list(t.State), gridcolor="rgba(0,0,0,0)",
                     tickfont=dict(family=FONT, size=11.5, color=SUBTLE))
    top, bot = t.iloc[-1], t.iloc[0]
    cap = (f"{top.State} does {top.gap:.0f} points better than its spending of {inr(top.pc)} per capita would "
           f"predict, the best value for money of any state, while {bot.State} does {abs(bot.gap):.0f} points worse "
           f"than expected on {inr(bot.pc)}. {int(better.sum())} of {len(t)} states beat their prediction. How "
           f"well money is used matters as well as how much is spent.")
    return fig, cap

def two_indices(b, sel):
    """Side by side: spending vs the Outcome Index and spending vs the Care Index, same states, same scale."""
    t = b.total
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=["Outcome Index", "Care Index"])
    for i, k in enumerate(["outcome_idx", "care_idx"], start=1):
        hov = (np.stack([t.State, t.pc.map(inr), t[k].round(0)], axis=1),
               "<b>%{customdata[0]}</b><br>%{customdata[1]} per capita<br>" + label_any(k)
               + " %{customdata[2]}<extra></extra>")
        _scatter_regions(fig, t, "pc", k, hover=hov, row=1, col=i, showlegend=i == 1, base=11)
        s = D.spearman(t.pc, t[k])
        sl, ic = D.log_fit(t.pc, t[k])
        xs = np.logspace(np.log10(t.pc.min() * 0.92), np.log10(t.pc.max() * 1.08), 40)
        # solid white trend line when the link is significant, faint dotted line when it is not
        fig.add_trace(go.Scatter(x=xs, y=ic + sl * np.log10(xs), mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color=TEXT if _sig(s["p"]) else NEUTRAL, width=2.2,
                                           dash="solid" if _sig(s["p"]) else "dot")), row=1, col=i)
        fig.add_annotation(x=0.02, y=0.98, xref=f"x{'' if i == 1 else i} domain", yref="y domain",
                           xanchor="left", yanchor="top", text=_rho_text(s), showarrow=False,
                           font=dict(family=FONT_MONO, size=12, color=TEXT), bgcolor=LABEL_BG, borderpad=5)
        _log_axis(fig, title="Health spending per capita (₹, log scale)", row=1, col=i, ticks=[500, 1000, 2000, 4000])
        if sel in set(t.State):
            r = t[t.State == sel].iloc[0]
            _highlight_point(fig, r.pc, r[k], sel if i == 1 else "", row=1, col=i, size=17)
    base_layout(fig, "Outcomes rise with spending, care does not",
                "Same 29 states in both panels · solid line = significant trend, dotted line = no clear trend",
                height=760)
    fig.update_yaxes(range=[-3, 103], title_text="Index (0 = worst state, 100 = best)", row=1, col=1)
    so, sc = D.spearman(t.pc, t.outcome_idx), D.spearman(t.pc, t.care_idx)
    rest = t[~t.flagged_ne]
    sc2 = D.spearman(rest.pc, rest.care_idx)
    cap = (f"In the left panel, the states climb with spending: higher-spending states have clearly healthier "
           f"mothers and children (ρ = {so['r']:+.2f}). In the right panel, however, they do not: how much antenatal care, "
           f"facility births and vaccination families receive is unrelated to spending (ρ = {sc['r']:+.2f}). ")
    if _sig(sc2["p"]) and not _sig(sc["p"]):
        cap += (f"The reason is partly the six small north-eastern states, which spend a lot but have little formal "
                f"care. Without them, care does rise with spending (ρ = {sc2['r']:+.2f}).")
    else:
        cap += "Whatever extra spending buys, it is not mainly more routine care."
    return fig, cap

# Tab 4 - Is it really the spending?
CONFOUNDER_KEYS = ["school", "lit", "san", "water", "insur"]
CONFOUNDER_SHORT = {"school": "women's schooling", "lit": "women's literacy", "san": "sanitation",
                    "water": "drinking water", "insur": "health insurance"}

def confounders(b):
    """Confounders with enough data to test."""
    return [k for k in CONFOUNDER_KEYS if k in D.IND and b.total[k].notna().sum() >= 10]

def _confounder_sentence(raw, part, sc, short):
    """Verdict on one possible confounder."""
    if sc["r"] <= -0.3:
        cap = (f"Higher-spending states actually tend to do worse on {short} (ρ = {sc['r']:+.2f}). Therefore, it cannot "
               f"be the hidden reason for these states to be healthier. ")
    elif sc["r"] < 0.3:
        cap = (f"{short[0].upper() + short[1:]} has little to do with a state's expenditure (ρ = {sc['r']:+.2f}), "
               f"so it cannot be the hidden reason that higher spenders are healthier. ")
    else:
        cap = (f"Higher-spending states also tend to do better on {short} (ρ = {sc['r']:+.2f}), which could be the "
               f"real reason their mothers and children are healthier. ")
    if part["r"] >= raw["r"]:
        cap += f"Allowing for it, the spending link does not weaken at all ({raw['r']:+.2f} to {part['r']:+.2f})."
    elif _sig(part["p"]):
        cap += (f"However, when states with similar {short} are compared, spending still matters: the link falls only "
                f"from {raw['r']:+.2f} to {part['r']:+.2f} ({fmt_p(part['p'])}), keeping about "
                f"{part['r'] / raw['r'] * 100:.0f}% of its strength.")
    else:
        cap += (f"Once {short} is allowed for, the spending link largely disappears ({raw['r']:+.2f} to "
                f"{part['r']:+.2f}, {fmt_p(part['p'])}).")
    return cap

def confounder_scatter(b, sel, conf="school"):
    """Spending vs Outcome Index, dots coloured by a possible confounder (say, women's schooling)"""
    t = b.total.dropna(subset=[conf])
    lab, unit, short = D.IND[conf].label, D.IND[conf].unit, CONFOUNDER_SHORT.get(conf, _lc(D.IND[conf].label))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t.pc, y=t.outcome_idx, mode="markers", showlegend=False,
        marker=dict(size=15, color=t[conf], colorscale=SEQ_ALT, line=dict(color=CARD, width=1),
                    colorbar=dict(title=dict(text=f"{short.capitalize()}<br>({unit})", font=dict(color=MUTED, size=11.5)),
                                  tickfont=dict(family=FONT_MONO, color=FAINT, size=11), thickness=10,
                                  len=0.6, outlinewidth=0)),
        customdata=np.stack([t.State, t.pc.map(inr), t.outcome_idx.round(0), t[conf].round(1)], axis=1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} per capita<br>Outcome Index %{customdata[2]}<br>"
                      + lab + ": %{customdata[3]}<extra></extra>"))
    if sel in set(t.State):
        r = t[t.State == sel].iloc[0]
        _highlight_point(fig, r.pc, r.outcome_idx, sel, size=22)
    raw = D.spearman(t.pc, t.outcome_idx)
    part = D.partial_spearman(t.pc, t.outcome_idx, t[conf])
    sc = D.spearman(t.pc, t[conf])
    base_layout(fig, f"Is it the money, or {short}?",
                f"Spending vs Outcome Index · dot colour = {short} (lighter = higher)",
                height=600, legend=False)
    _log_axis(fig)
    fig.update_yaxes(title_text="Outcome Index (0 = worst state, 100 = best)")
    fig.add_annotation(x=0.99, y=0.02, xref="paper", yref="paper", xanchor="right", yanchor="bottom",
                       showarrow=False, font=dict(family=FONT_MONO, size=12, color=TEXT), bgcolor=LABEL_BG,
                       bordercolor=AXIS, borderpad=6, align="right",
                       text=f"ρ {raw['r']:+.2f} → {part['r']:+.2f}<br>after allowing for {short}")
    return fig, _confounder_sentence(raw, part, sc, short)

def confounder_summary(b, sel):
    """Dumbbell: the spending-outcome link before and after allowing for each possible confounder."""
    t = b.total
    raw = D.spearman(t.pc, t.outcome_idx)
    rows = []
    for k in confounders(b):
        part = D.partial_spearman(t.pc, t.outcome_idx, t[k])
        rows.append(dict(label=D.IND[k].label, short=CONFOUNDER_SHORT.get(k, _lc(D.IND[k].label)),
                         r=part["r"], p=part["p"], with_spend=D.spearman(t.pc, t[k])["r"]))
    r = pd.DataFrame(rows).sort_values("r").reset_index(drop=True)
    fig = go.Figure()
    for _, x in r.iterrows():
        fig.add_trace(go.Scatter(x=[raw["r"], x.r], y=[x.label, x.label], mode="lines",
                                 line=dict(color=AXIS, width=3), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[raw["r"]] * len(r), y=r.label, mode="markers", name="Spending alone",
                             marker=dict(size=12, color=NEUTRAL, line=dict(color=CARD, width=1)),
                             hovertemplate=f"Spending alone: ρ = {raw['r']:+.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=r.r, y=r.label, mode="markers+text", name="After allowing for this factor",
        marker=dict(size=14, color=SERIES_A, symbol="diamond", line=dict(color=CARD, width=1)),
        text=[f"{v:+.2f}" for v in r.r],
        textposition=["middle right" if v >= raw["r"] else "middle left" for v in r.r],
        textfont=dict(family=FONT_MONO, color=SUBTLE, size=11),
        customdata=np.stack([r.p.map(fmt_p), r.with_spend.round(2)], axis=1),
        hovertemplate="<b>%{y}</b><br>After allowing for it: ρ = %{x:+.2f}, %{customdata[0]}"
                      "<br>Its own link with spending: ρ = %{customdata[1]}<extra></extra>"))
    n_sig = int((r.p < 0.05).sum())
    title = ("The spending link survives every other explanation tested" if n_sig == len(r)
             else f"The spending link survives {n_sig} of {len(r)} other explanations")
    base_layout(fig, title, "Spending vs Outcome Index (Spearman ρ), before and after allowing for each factor",
                height=600, margin=dict(l=210))
    fig.update_xaxes(title_text="Spearman ρ between spending and Outcome Index", range=[0, 1.0], dtick=0.2,
                     tickformat=".1f")
    fig.update_yaxes(categoryorder="array", categoryarray=list(r.label), gridcolor="rgba(0,0,0,0)",
                     tickfont=dict(family=FONT, size=11.5, color=SUBTLE))
    w = r.iloc[0]
    verdict = ("The relationship remains statistically significant in all checks" if n_sig == len(r)
           else f"The relationship remains statistically significant in {n_sig} of {len(r)} checks")
    cap = (f"States that spend more may also have advantages such as better education or cleaner water, "
       f"which could partly explain their better health outcomes. To examine this, "
       f"{len(r)} potential influencing factors were tested, one at a time. {verdict}. The correlation (ρ) "
       f"between spending and health outcomes is {raw['r']:+.2f} before adjustment and ranges "
       f"from {w.r:+.2f} to {r.r.max():+.2f} after accounting for these factors. The weakest "
       f"association occurs when {w.short} is taken into account.")
    return fig, cap

# NFHS-5 Phase 1 states: surveyed from June 2019 to January 2020 before COVID-19 hit
PHASE1 = ["Andhra Pradesh", "Assam", "Bihar", "Goa", "Gujarat", "Himachal Pradesh", "Jammu and Kashmir",
          "Karnataka", "Kerala", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Sikkim",
          "Telangana", "Tripura", "West Bengal"]
SENSITIVITY_KEYS = ["outcome_idx", "care_idx", "imr", "u5mr", "stunt", "anaem_child", "anaem_preg", "inst", "vacc", "oop"]

def sensitivity(b, sel):
    """Signed ρ for key indicators: all states vs without the six NE states vs pre-COVID (Phase 1) states only."""
    t = b.total
    groups = [("all", "All 29 states", t, SERIES_A, "circle"),
              ("wo", f"Without the 6 north-eastern states ({int((~t.flagged_ne).sum())})", t[~t.flagged_ne],
               SERIES_B, "diamond"),
              ("p1", f"Surveyed before COVID only ({int(t.State.isin(PHASE1).sum())})", t[t.State.isin(PHASE1)],
               TEXT, "square-open")]
    rows = []
    for k in SENSITIVITY_KEYS:
        row = dict(key=k, label=label_any(k))
        for g, _, d, _, _ in groups:
            sp = D.spearman(d.pc, d[k])
            row[g], row["p_" + g] = sp["r"] * better_any(k), sp["p"]
        rows.append(row)
    r = pd.DataFrame(rows)[::-1].reset_index(drop=True)
    fig = go.Figure()
    xs, ys = [], []
    for _, x in r.iterrows():
        lo, hi = min(x["all"], x["wo"], x["p1"]), max(x["all"], x["wo"], x["p1"])
        xs += [lo, hi, None]
        ys += [x.label, x.label, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=AXIS, width=3), hoverinfo="skip",
                             showlegend=False))
    for g, name, _, col, sym in groups:
        fig.add_trace(go.Scatter(
            x=r[g], y=r.label, mode="markers", name=name,
            marker=dict(size=12, color=col, symbol=sym, line=dict(color=CARD if "open" not in sym else col, width=1.5)),
            customdata=r["p_" + g].map(fmt_p),
            hovertemplate="<b>%{y}</b><br>" + name + ": ρ = %{x:+.2f}, %{customdata}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1.5))
    fig.add_annotation(x=0.01, y=1.0, xref="paper", yref="paper", xanchor="left", yanchor="bottom", showarrow=False,
                       text="◀ worse with more spending", font=dict(size=11, color=FAINT))
    fig.add_annotation(x=0.99, y=1.0, xref="paper", yref="paper", xanchor="right", yanchor="bottom", showarrow=False,
                       text="better with more spending ▶", font=dict(size=11, color=FAINT))
    base_layout(fig, "Do unusual states or COVID change the result?",
                "Spearman ρ between spending and each indicator, recalculated on three groups of states",
                height=600, margin=dict(l=200))
    fig.update_xaxes(range=[-1, 1], dtick=0.25, tickformat="+.2f", title_text="Spearman ρ (+ = better with more spending)")
    fig.update_yaxes(tickfont=dict(family=FONT, size=11.5, color=SUBTLE), gridcolor="rgba(0,0,0,0)")
    # Plain-language caption
    o = r[r.key == "outcome_idx"].iloc[0]
    c = r[r.key == "care_idx"].iloc[0]
    held = [g for g in ("wo", "p1") if o[g] > 0 and _sig(o["p_" + g])]
    n_p1 = int(t.State.isin(PHASE1).sum())
    cap = (f"The main result is not driven by a few unusual states or by the pandemic: higher spending goes with "
           f"healthier mothers and children in all three groups (ρ = {o['all']:+.2f} for all states, "
           f"{o['wo']:+.2f} without the north-east and {o['p1']:+.2f} for the {n_p1} states surveyed before COVID)"
           + ("." if len(held) == 2 else ", although the smaller groups make it less certain."))
    if c["wo"] - c["all"] > 0.3:
        cap += (f" Care, on the other hand, behaves differently: once the north-east is left out, higher spending does go with more "
                f"care (ρ = {c['all']:+.2f} → {c['wo']:+.2f}), likely because those states spend a lot but have "
                f"low use of hospitals and clinics.")
    return fig, cap

def oop_scatter(b, sel):
    """What families pay per delivery in a public facility vs what the state spends per capita."""
    t = b.total.dropna(subset=["oop"])
    s = D.spearman(t.pc, t.oop)
    fig = go.Figure()
    hov = (np.stack([t.State, t.pc.map(inr), t.oop.map(inr)], axis=1),
           "<b>%{customdata[0]}</b><br>State spends %{customdata[1]} per capita"
           "<br>Family pays %{customdata[2]} per delivery<extra></extra>")
    _scatter_regions(fig, t, "pc", "oop", hover=hov, base=13)
    sl, ic = D.log_fit(t.pc, t.oop)
    xs = np.logspace(np.log10(t.pc.min() * 0.92), np.log10(t.pc.max() * 1.08), 40)
    fig.add_trace(go.Scatter(x=xs, y=ic + sl * np.log10(xs), mode="lines", name="Trend",
                             line=dict(color=TEXT if _sig(s["p"]) else NEUTRAL,
                                       dash="solid" if _sig(s["p"]) else "dot", width=2), hoverinfo="skip"))
    _label_points(fig, t, "pc", "oop",
                  [x for x in list(t.nlargest(3, "oop").State) + list(t.nsmallest(1, "oop").State) if x != sel],
                  logx=True)
    if sel in set(t.State):
        r = t[t.State == sel].iloc[0]
        _highlight_point(fig, r.pc, r.oop, sel, size=20)
    title = ("Families pay more per delivery where states spend more" if s["r"] > 0
             else "Families pay less per delivery where states spend more")
    base_layout(fig, title,
                "Average out-of-pocket cost of a delivery in a public facility vs state health spending per capita",
                height=600)
    _log_axis(fig)
    fig.update_yaxes(title_text="Out-of-pocket cost per delivery (₹)", tickprefix="₹", tickformat=",")
    w = D.spearman(t[~t.flagged_ne].pc, t[~t.flagged_ne].oop)
    top, low = t.loc[t.oop.idxmax()], t.loc[t.oop.idxmin()]
    expected = s["r"] < 0
    cap = (f"One might expect families to pay less where the state spends more, "
           f"{'and that is what happens' if expected else 'but the opposite happens'} "
           f"(ρ = {s['r']:+.2f}, {fmt_p(s['p'])}): a delivery in a public facility costs families "
           f"{inr(low.oop)} in {low.State} but {inr(top.oop)} in {top.State}. "
           f"{'This still holds' if _sig(w['p']) and np.sign(w['r']) == np.sign(s['r']) else 'This weakens'} "
           f"without the north-eastern states (ρ = {w['r']:+.2f}). A possible reason is that care costs more in "
           f"remote and hilly states, although this claim cannot be backed up by the available data.")
    return fig, cap