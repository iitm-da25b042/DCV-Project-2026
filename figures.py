"""Chart functions for the dashboard"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import data as D
from theme import (AXIS, CARD, FAINT, FONT, FONT_MONO, GRID, HL, MUTED, NEUTRAL, REGION_COLOURS, REGION_SYMBOLS, SEQ, SEQ_ALT, SEQ_COUNT, SERIES_A, SERIES_B, SUBTLE, TEXT, base_layout, region_legend_traces)

inr = D.inr
fmt_p = D.fmt_p
LOG_TICKS = [500, 1000, 2000, 4000, 8000]   # ₹ ticks for log-scale spending axes
LABEL_BG = "rgba(0,0,0,0.75)"

# Helpers
def _log_axis(fig, title="Health spending per capita (₹, log scale)", row=None, col=None, ticks=LOG_TICKS):
    """Log-scale x axis with ₹ tick labels"""
    kw = dict(type="log", title_text=title, tickvals=ticks, ticktext=[inr(t) for t in ticks])
    if row:
        fig.update_xaxes(row=row, col=col, **kw)
    else:
        fig.update_xaxes(**kw)
def _lc(label):
    """Converts a label into lower case for use mid-sentence, while retaining acronyms"""
    return " ".join(w if (len(w) > 1 and w[:2].isupper()) else w.lower() for w in str(label).split())
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
def _scatter_regions(fig, d, x, y, size=None, hover=None, row=None, col=None, showlegend=True,
                     imputed_col=None, sizeref=None, base=11):
    """One trace for each region: colour and marker shape, with imputed points being drawn as hollow."""
    for r in D.REGION_ORDER:
        mask = (d.Region == r).values
        g = d[mask]
        if g.empty:
            continue
        flags = g[imputed_col] if imputed_col else [False] * len(g)
        sym = [REGION_SYMBOLS[r] + ("-open" if v else "") for v in flags]
        marker = dict(color=REGION_COLOURS[r], symbol=sym, line=dict(color=CARD, width=1.2), opacity=0.92)
        if size is not None:
            marker.update(size=g[size], sizemode="area", sizeref=sizeref, sizemin=7)
        else:
            marker.update(size=base)
        tr = go.Scatter(x=g[x], y=g[y], mode="markers", name=r, legendgroup=r, showlegend=showlegend,
                        marker=marker, customdata=hover[0][mask] if hover else None,
                        hovertemplate=hover[1] if hover else None, text=g["State"])
        if row:
            fig.add_trace(tr, row=row, col=col)
        else:
            fig.add_trace(tr)
def _highlight_point(fig, x, y, label, row=None, col=None, size=15, textpos="top center"):
    """Rings the selected state in the highlight colour and labels it."""
    tr = go.Scatter(x=[x], y=[y], mode="markers+text", text=[f"<b>{label}</b>"], textposition=textpos,
                    textfont=dict(color=HL, size=12.5), showlegend=False, hoverinfo="skip",
                    marker=dict(size=size, color="rgba(0,0,0,0)", line=dict(color=HL, width=2.5)))
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
        fx = (px - xr[0]) / ((xr[1] - xr[0]) or 1)   # position as a fraction of the axis range
        fy = (py - yr[0]) / ((yr[1] - yr[0]) or 1)
        ay = (24 if fy > 0.85 else -24) - (14 if i % 2 else 0) * (1 if fy <= 0.85 else -1)
        ax = -30 if fx > 0.8 else (30 if fx < 0.12 else [0, -34, 34][i % 3])
        fig.add_annotation(x=px, y=py, text=nm, showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=FAINT,
                           ax=ax, ay=ay, font=dict(size=11.5, color=SUBTLE), bgcolor=LABEL_BG,
                           xanchor="right" if ax < 0 else ("left" if ax > 0 else "center"))

# Header KPI cards
def kpis(b):
    """Headline numbers shown in cards above the tabs."""
    t = b.total
    lo, hi = t.loc[t.pc.idxmin()], t.loc[t.pc.idxmax()]
    so = D.spearman(t.pc, t.outcome_idx)
    sc = D.spearman(t.pc, t.care_idx)
    si = D.spearman(t.pc, t.imr)
    return [
        dict(label="Median health spending per capita", value=inr(t.pc.median()),
             note=f"From {inr(lo.pc)} ({lo.State}) to {inr(hi.pc)} ({hi.State})"),
        dict(label="Spending vs Outcome Index", value=f"ρ = {so['r']:+.2f}", accent=True,
             note=f"Higher spenders have healthier mothers and children ({fmt_p(so['p'])})"),
        dict(label="Spending vs Care Index", value=f"ρ = {sc['r']:+.2f}",
             note=f"No clear link with care received ({fmt_p(sc['p'])})"),
        dict(label="Spending vs infant mortality", value=f"ρ = {si['r']:+.2f}",
             note=f"Higher spenders lose fewer babies in the first year ({fmt_p(si['p'])})"),
    ]

# Tab 1: Spending Landscape
def map_spending(b, sel):
    """Choropleth of spending per capita (log colour scale). Grey states are not in the study."""
    t = b.total
    names = [f["properties"]["st_nm"] for f in b.geojson["features"]]
    nodata = [n for n in names if n not in set(t.State)]
    fig = go.Figure()
    # Layer 1: States/UTs not in the study, in flat grey
    fig.add_trace(go.Choropleth(
        geojson=b.geojson, featureidkey="properties.st_nm", locations=nodata, z=[0] * len(nodata),
        colorscale=[[0, GRID], [1, GRID]], showscale=False, marker_line_color=AXIS,
        marker_line_width=0.5, hovertemplate="%{location}<br>No data in this study<extra></extra>"))
    # Layer 2: 29 states, coloured by spending per capita
    ticks = [500, 1000, 2000, 4000]
    fig.add_trace(go.Choropleth(
        geojson=b.geojson, featureidkey="properties.st_nm", locations=t.State, z=np.log10(t.pc),
        colorscale=SEQ, zmin=np.log10(450), zmax=np.log10(6500), marker_line_color=CARD,
        marker_line_width=0.7,
        customdata=np.stack([t.pc.map(inr), t.spend_cr.map(inr), t.pop_m.round(1), t.Region], axis=1),
        hovertemplate="<b>%{location}</b> (%{customdata[3]})<br>%{customdata[0]} per capita"
                      "<br>Total: %{customdata[1]} crore<br>Population: %{customdata[2]} million<extra></extra>",
        colorbar=dict(title=dict(text="₹ per capita", side="top", font=dict(color=MUTED, size=12)),
                      orientation="h", x=0.5, xanchor="center", y=-0.02, yanchor="top",
                      tickvals=np.log10(ticks), ticktext=[inr(v) for v in ticks],
                      tickfont=dict(family=FONT_MONO, color=FAINT, size=11), thickness=10, len=0.45,
                      outlinewidth=0)))
    # Layer 3: Outline the selected state in the highlight colour
    if sel in set(t.State):
        fig.add_trace(go.Choropleth(
            geojson=b.geojson, featureidkey="properties.st_nm", locations=[sel], z=[1],
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]], showscale=False,
            marker_line_color=HL, marker_line_width=3, hoverinfo="skip"))
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)",
                    projection_type="mercator")
    base_layout(fig, "Health spending per capita by state",
                "Actual state health expenditure (Accounts) ÷ Population · Click a state to highlight it",
                height=680, legend=False, margin=dict(l=8, r=8, b=70))
    gap = t.pc.max() / t.pc.min()
    cap = (f"Per capita expenditure ranges from {inr(t.pc.min())} in {t.loc[t.pc.idxmin(), 'State']} to "
           f"{inr(t.pc.max())} in {t.loc[t.pc.idxmax(), 'State']}, a {gap:.0f}-fold gap. "
           f"The brightest states are mostly small hilly and north-eastern states, while the large "
           f"plains states of the north and east sit at the dark end; the median is {inr(t.pc.median())}.")
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
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=AXIS, width=2),
                             hoverinfo="skip", showlegend=False))
    # Region-coloured, region-shaped heads
    for reg in D.REGION_ORDER:
        g = t[t.Region == reg]
        fig.add_trace(go.Scatter(
            x=g.pc, y=g.State, mode="markers", name=reg,
            marker=dict(color=REGION_COLOURS[reg], symbol=REGION_SYMBOLS[reg], size=13,
                        line=dict(color=CARD, width=1)),
            customdata=np.stack([g.pc.map(inr), g.pop_m.round(1)], axis=1),
            hovertemplate="<b>%{y}</b><br>%{customdata[0]} per capita"
                          "<br>%{customdata[1]} million people<extra></extra>"))
    # Median acts as the reference line
    fig.add_vline(x=med, line=dict(color=FAINT, width=1.2, dash="dash"))
    fig.add_annotation(x=med, y=0, yref="paper", text=f"Median {inr(med)}", showarrow=False,
                       xanchor="left", yanchor="bottom", xshift=6, yshift=4,
                       font=dict(color=SUBTLE, size=11.5))
    if sel in set(t.State):
        r = t[t.State == sel].iloc[0]
        _highlight_point(fig, r.pc, r.State, f"  {inr(r.pc)}", textpos="middle right")
    lo, hi = t.iloc[0], t.iloc[-1]
    base_layout(fig, "Health spending per capita, ranked by state",
                f"Coloured and shaped by region · {hi.State} spends {hi.pc / lo.pc:.0f}× more per capita "
                f"than {lo.State}", height=760)
    fig.update_xaxes(title_text="Health spending per capita (₹)", tickprefix="₹", tickformat=",",
                     rangemode="tozero")
    fig.update_yaxes(categoryorder="array", categoryarray=list(t.State), gridcolor="rgba(0,0,0,0)",
                     tickfont=dict(family=FONT, size=11.5, color=SUBTLE))
    top = t.nlargest(8, "pc")
    small = int((top.pop_m < 15).sum())
    cap = (f"Half the states spend less than {inr(med)} per capita. "
           f"{'All' if small == 8 else small} of the 8 highest spenders have fewer than 15 million people, and "
           f"{(top.Region == 'Northeast').sum()} of those 8 are in the Northeast, where small populations "
           f"and hard terrain raise the cost of reaching each person.")
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
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[r.r_total, r.r_pc], mode="lines+markers",
            line=dict(color=col, width=3.5 if is_sel else 1.3), opacity=1 if is_sel or not sel else 0.5,
            marker=dict(size=9 if is_sel else 7, color=col, symbol=REGION_SYMBOLS[r.Region]),
            name=r.Region, legendgroup=r.Region, showlegend=False,
            customdata=[[r.State, inr(r.spend_cr), inr(r.pc)]] * 2,
            hovertemplate="<b>%{customdata[0]}</b><br>Total %{customdata[1]} crore"
                          "<br>%{customdata[2]} per capita<extra></extra>"))
    # State names and ranks on both sides
    for _, r in t.iterrows():
        c = HL if r.State == sel else SUBTLE
        fig.add_annotation(x=0, y=r.r_total, text=f"{r.State}  {r.r_total}", xanchor="right", xshift=-8,
                           showarrow=False, font=dict(size=11, color=c))
        fig.add_annotation(x=1, y=r.r_pc, text=f"{r.r_pc}  {r.State}", xanchor="left", xshift=8,
                           showarrow=False, font=dict(size=11, color=c))
    region_legend_traces(fig, [r for r in D.REGION_ORDER if r in set(t.Region)])
    base_layout(fig, "A large budget does not imply high per capita expenditure",
                "Rank by total health spending (left) vs rank by spending per capita (right) · 1 = highest",
                height=640, margin=dict(l=150, r=150))
    fig.update_xaxes(visible=False, range=[-0.05, 1.05])
    fig.update_yaxes(autorange="reversed", visible=False)
    up = t.loc[t.move.idxmax()]
    down = t.loc[t.move.idxmin()]
    cap = (f"{down.State} has the {_ordinal(down.r_total)}-largest health budget in total but ranks "
           f"{_ordinal(down.r_pc)} per capita, while {up.State} jumps from {_ordinal(up.r_total)} to "
           f"{_ordinal(up.r_pc)}. Totals mostly measure state size, which is why every spending comparison is reported in "
           f"per capita terms.")
    return fig, cap

def treemap_spending(b, sel):
    """Region > state treemap: box size = total spending, colour = spending per capita."""
    t = b.total
    reg = (t.groupby("Region")
           .agg(spend_cr=("spend_cr", "sum"), pop=("Population (Total)", "sum"))
           .reset_index())
    reg["pc"] = reg.spend_cr * 1e7 / reg["pop"]
    # Three levels: India -> regions -> states
    ids = ["India"] + list(reg.Region) + list(t.Region + "/" + t.State)
    labels = ["India"] + list(reg.Region) + list(t.State)
    parents = [""] + ["India"] * len(reg) + list(t.Region)
    values = [t.spend_cr.sum()] + list(reg.spend_cr) + list(t.spend_cr)
    pcs = [t.spend_cr.sum() * 1e7 / t["Population (Total)"].sum()] + list(reg.pc) + list(t.pc)
    # Thick highlight-coloured border for selected state
    line_cols = [CARD] * (1 + len(reg)) + [HL if s == sel else CARD for s in t.State]
    line_w = [1] * (1 + len(reg)) + [4 if s == sel else 1 for s in t.State]
    ticks = [500, 1000, 2000, 4000]
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
        marker=dict(colors=np.log10(pcs), colorscale=SEQ, cmin=np.log10(450), cmax=np.log10(6500),
                    line=dict(color=line_cols, width=line_w),
                    colorbar=dict(title=dict(text="₹ per capita", side="top",
                                             font=dict(color=MUTED, size=12)),
                                  orientation="h", x=0.5, xanchor="center", y=-0.02, yanchor="top",
                                  tickvals=np.log10(ticks), ticktext=[inr(v) for v in ticks],
                                  tickfont=dict(family=FONT_MONO, color=FAINT, size=11),
                                  thickness=10, len=0.45, outlinewidth=0)),
        customdata=np.stack([[inr(v) for v in values], [inr(v) for v in pcs]], axis=1),
        texttemplate="<b>%{label}</b><br>%{customdata[1]} per capita",
        hovertemplate="<b>%{label}</b><br>Total %{customdata[0]} crore<br>%{customdata[1]} per capita"
                      "<br>%{percentRoot:.1%} of all spending<extra></extra>",
        textfont=dict(family=FONT, size=12), root_color="rgba(0,0,0,0)", tiling=dict(pad=2),
        pathbar=dict(visible=False)))
    fig.update_layout(uniformtext=dict(minsize=9, mode="hide"))   # Hide labels that do not fit
    base_layout(fig, "Where is the health money and how thinly is it spread?",
                "Box size = total health spending (₹ crore) · Colour = spending per capita · "
                "Click a region to zoom",
                height=680, legend=False, margin=dict(l=8, r=8, b=70))
    top5 = t.nlargest(5, "spend_cr")
    share_spend = top5.spend_cr.sum() / t.spend_cr.sum() * 100
    share_pop = top5["Population (Total)"].sum() / t["Population (Total)"].sum() * 100
    ne = reg[reg.Region == "Northeast"].iloc[0]
    cap = (f"The five biggest budgets ({', '.join(top5.State)}) make up {share_spend:.0f}% of all state "
           f"health spending but serve {share_pop:.0f}% of the population. The boxes for the Northeastern states are small "
           f"but bright: together this region spends {inr(ne.pc)} per capita, against {inr(pcs[0])} for all 29 "
           f"states combined.")
    return fig, cap

def spending_histogram(b, sel):
    """Distribution of spending per capita: histogram, a rug of ticks (one per state), median and mean."""
    t = b.total
    fig = go.Figure()
    # Bars coloured by how many states they hold (taller = brighter), with a colour bar as the key
    edges = np.arange(0, 7000, 500)
    counts, _ = np.histogram(t.pc, bins=edges)
    mids = (edges[:-1] + edges[1:]) / 2
    fig.add_trace(go.Bar(
        x=mids, y=counts, width=480, showlegend=False,
        marker=dict(color=counts, colorscale=SEQ_COUNT, cmin=0, cmax=int(counts.max()), line=dict(width=0),
                    colorbar=dict(title=dict(text="Number<br>of states", font=dict(color=MUTED, size=12)),
                                  tickvals=list(range(0, int(counts.max()) + 1, 2)),
                                  tickfont=dict(family=FONT_MONO, color=FAINT, size=11),
                                  thickness=10, len=0.55, y=0.62, outlinewidth=0)),
        customdata=[f"{inr(lo)}–{inr(lo + 500)}" for lo in edges[:-1]],
        hovertemplate="%{customdata}<br>%{y} states<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=t.pc, y=[-0.6] * len(t), mode="markers", showlegend=False, text=t.State,
        marker=dict(symbol="line-ns", size=16, line=dict(color=SUBTLE, width=1.6)),
        customdata=t.pc.map(inr), hovertemplate="<b>%{text}</b><br>%{customdata} per capita<extra></extra>"))
    med, mean = t.pc.median(), t.pc.mean()
    fig.add_vline(x=med, line=dict(color=FAINT, width=1.2, dash="dash"))
    fig.add_annotation(x=med, y=1, yref="paper", text=f"Median {inr(med)}", showarrow=False,
                       xanchor="left", xshift=4, yanchor="top", font=dict(color=SUBTLE, size=11.5))
    fig.add_vline(x=mean, line=dict(color=FAINT, width=1.2, dash="dot"))
    fig.add_annotation(x=mean, y=0.9, yref="paper", text=f"Mean {inr(mean)}", showarrow=False,
                       xanchor="left", xshift=4, yanchor="top", font=dict(color=FAINT, size=11.5))
    if sel in set(t.State):
        r = t[t.State == sel].iloc[0]
        fig.add_trace(go.Scatter(
            x=[r.pc], y=[-0.6], mode="markers+text", text=[f"<b>{sel}</b>"], textposition="bottom center",
            textfont=dict(color=HL, size=12), showlegend=False, hoverinfo="skip",
            marker=dict(symbol="line-ns", size=22, line=dict(color=HL, width=3))))
    base_layout(fig, "Most states cluster at the low end",
                "Number of states in each ₹500 band of spending per capita · ticks below show each state",
                height=640, legend=False)
    fig.update_xaxes(title_text="Health spending per capita (₹)", tickprefix="₹", tickformat=",")
    fig.update_yaxes(title_text="Number of states", range=[-1.6, None], dtick=2)
    n15 = int((t.pc < 1500).sum())
    n4k = int((t.pc > 4000).sum())
    cap = (f"{n15} of {len(t)} states spend under ₹1,500 per capita, while a long tail of {n4k} states "
           f"spends over ₹4,000. Since the mean ({inr(mean)}) is pulled well above the median "
           f"({inr(med)}), rank-based correlations (Spearman) and log scales are used.")
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
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=AXIS, width=2),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=t.util_re, y=t.State, mode="markers", name="Revised Estimate",
        marker=dict(symbol="diamond-open", size=11, color=SERIES_A, line=dict(width=2)),
        customdata=t.pc_re.map(inr),
        hovertemplate="<b>%{y}</b><br>Revised Estimate: %{x:.1f}% of budget"
                      "<br>(%{customdata} per capita)<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=t.util_be, y=t.State, mode="markers", name="Actual spending (Accounts)",
        marker=dict(symbol="circle", size=11, color=SERIES_B, line=dict(color=CARD, width=1)),
        customdata=np.stack([t.pc.map(inr), t.pc_be.map(inr)], axis=1),
        hovertemplate="<b>%{y}</b><br>Actual: %{x:.1f}% of budget"
                      "<br>%{customdata[0]} spent vs %{customdata[1]} budgeted per capita<extra></extra>"))
    fig.add_vline(x=100, line=dict(color=FAINT, width=1.2))
    fig.add_annotation(x=100, y=0, yref="paper", text="100% = spent as budgeted", showarrow=False,
                       xanchor="right", yanchor="bottom", xshift=-6, yshift=4,
                       font=dict(color=SUBTLE, size=11.5), bgcolor=LABEL_BG)
    if sel in set(t.State):
        r = t[t.State == sel].iloc[0]
        _highlight_point(fig, r.util_be, r.State, f"  {r.util_be:.0f}%", textpos="middle right")
    base_layout(fig, "Most states spent less on health than they budgeted",
                "2019-20 actual spending and Revised Estimate as % of the Budget Estimate · "
                "states in the same order as the spending ranking",
                height=760)
    fig.update_xaxes(title_text="% of Budget Estimate (2019-20)", ticksuffix="%")
    fig.update_yaxes(categoryorder="array", categoryarray=list(t.State), gridcolor="rgba(0,0,0,0)",
                     tickfont=dict(family=FONT, size=11.5, color=SUBTLE))
    under = t[t.util_be < 100]
    lo, hi = t.loc[t.util_be.idxmin()], t.loc[t.util_be.idxmax()]
    cap = (f"In 2019-20, {len(under)} of {len(t)} states spent less on health than they budgeted, by a "
           f"median of {100 - under.util_be.median():.0f}%. {lo.State} spent only {lo.util_be:.0f}% of its "
           f"Budget Estimate, while {hi.State} spent {hi.util_be:.0f}%. Budgets therefore "
           f"overstate the actual expenditure on healthcare in several states.")
    return fig, cap