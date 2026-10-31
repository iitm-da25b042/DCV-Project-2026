"""Theme For The Dashboard: Colours, Fonts and Layout"""

import textwrap
import plotly.graph_objects as go

BG = "#000000"
CARD = "#171A26"
GRID = "#232A37"
AXIS = "#3A4354"
TEXT = "#E9ECF2"
SUBTLE = "#C4CAD6"
MUTED = "#A3ACBC"
FAINT = "#8C96A8"
NEUTRAL = "#6E7A91"
HL = "#2EE6FF"

# Adjacent Pairs
SERIES_A = "#3987e5"
SERIES_B = "#d95926"

# Regions
REGION_COLOURS = {
    "South": "#3987e5",
    "North": "#d95926",
    "East": "#199e70",
    "Central": "#9085e9",
    "West": "#d55181",
    "Northeast": "#8a9a2a",
    "Other": NEUTRAL,
}
REGION_SYMBOLS = {
    "South": "circle",
    "North": "square",
    "East": "diamond",
    "Central": "triangle-up",
    "West": "pentagon",
    "Northeast": "star",
    "Other": "circle",
}

# Colour Scales
SEQ = "Viridis"
SEQ_ALT = "Cividis"

# Fonts
FONT = "Inter, system-ui, sans-serif"
FONT_MONO = "JetBrains Mono, ui-monospace, monospace"
FONT_DISPLAY = "Outfit, Inter, sans-serif"


def base_layout(
    fig: go.Figure,
    title: str,
    subtitle: str = "",
    height: int = 460,
    legend: bool = True,
    margin=None,
    wrap: int = 96,
) -> go.Figure:
    """Title and wrapped subtitle at the top left with legend in its own band under them"""
    t = f"<b>{title}</b>"
    lines = textwrap.wrap(subtitle, wrap) if subtitle else []
    for ln in lines:
        t += f"<br><span style='font-size:12.5px;color:{MUTED};font-family:{FONT}'>{ln}</span>"
    title_px = 40 + 19 * len(lines)
    has_sub_titles = any(
        a.yanchor == "bottom" and a.yref == "paper" and (a.y or 0) >= 0.99
        for a in (fig.layout.annotations or [])
    )
    top = title_px + (40 if legend else 16) + (24 if has_sub_titles else 0)
    m = dict(l=64, r=24, t=top, b=56)
    if margin:
        m.update(margin)
    fig.update_layout(
        title=dict(
            text=t,
            x=0.0,
            xanchor="left",
            xref="container",
            y=1.0,
            yanchor="top",
            yref="container",
            font=dict(family=FONT_DISPLAY, size=17, color=TEXT),
            pad=dict(l=6, t=12),
        ),
        height=height,
        margin=m,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12.5, color=SUBTLE),
        showlegend=legend,
        legend=dict(
            orientation="h",
            yanchor="top",
            yref="container",
            y=1 - (title_px + 4) / height,
            xanchor="left",
            x=0,
            font=dict(size=12, color=SUBTLE),
            bgcolor="rgba(0,0,0,0)",
            itemsizing="constant",
            title_text="",
        ),
        hoverlabel=dict(
            bgcolor="#0B0E14",
            bordercolor=AXIS,
            font=dict(family=FONT, size=12.5, color=TEXT),
        ),
        hovermode="closest",
    )
    fig.update_xaxes(
        gridcolor=GRID,
        zerolinecolor=AXIS,
        linecolor=AXIS,
        tickcolor=AXIS,
        tickfont=dict(family=FONT_MONO, size=11.5, color=FAINT),
        title_font=dict(size=12.5, color=MUTED),
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=AXIS,
        linecolor=AXIS,
        tickcolor=AXIS,
        tickfont=dict(family=FONT_MONO, size=11.5, color=FAINT),
        title_font=dict(size=12.5, color=MUTED),
        automargin=True,
    )
    for a in fig.layout.annotations or []:  # subplot titles
        if a.font.size is None:
            a.font = dict(family=FONT_DISPLAY, size=13.5, color=SUBTLE)
    return fig


def region_legend_traces(fig, regions, row=None, col=None):
    """Adds legend only entries (colour and symbol) for each region"""
    for r in regions:
        tr = go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=r,
            legendgroup=r,
            marker=dict(
                color=REGION_COLOURS[r],
                symbol=REGION_SYMBOLS[r],
                size=11,
                line=dict(color=CARD, width=1),
            ),
        )
        if row:
            fig.add_trace(tr, row=row, col=col)
        else:
            fig.add_trace(tr)
