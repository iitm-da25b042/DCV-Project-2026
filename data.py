"""Data loading, derived variables and statistics for the dashboard"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# Paths and constants
HERE = Path(__file__).parent
RAW_PATH = HERE / "data" / "raw" / "budget_nfhs_combined.csv"
CLEAN_PATH = HERE / "data" / "clean" / "budget_nfhs_combined_clean.csv"
GEOJSON_PATH = HERE / "data" / "geo" / "india_states.geojson"

AREAS = ["Total", "Urban", "Rural"]
BASE_COLS = [
    "State",
    "Population (Total)",
    "Accounts",
    "Budget Estimates",
    "Revised Estimates",
]

# Budgets are in ₹ lakh
LAKH = 1e5

# Correcting wrong spellings of states to match the India map
STATE_FIXES = {
    "Chattisgarh": "Chhattisgarh",
    "Uttarkhand": "Uttarakhand",
    "Orissa": "Odisha",
}

# Classification of states into regions
REGIONS = {
    "South": [
        "Andhra Pradesh",
        "Karnataka",
        "Kerala",
        "Tamil Nadu",
        "Telangana",
    ],
    "North": [
        "Haryana",
        "Himachal Pradesh",
        "Jammu and Kashmir",
        "Punjab",
        "Rajasthan",
        "Uttarakhand",
    ],
    "East": ["Bihar", "Jharkhand", "Odisha", "West Bengal"],
    "Central": ["Chhattisgarh", "Madhya Pradesh", "Uttar Pradesh"],
    "West": ["Goa", "Gujarat", "Maharashtra"],
    "Northeast": [
        "Arunachal Pradesh",
        "Assam",
        "Manipur",
        "Meghalaya",
        "Mizoram",
        "Nagaland",
        "Sikkim",
        "Tripura",
    ],
}
REGION_OF = {s: r for r, ss in REGIONS.items() for s in ss}
REGION_ORDER = list(REGIONS)

# North East states flagged as outliers by DBSCAN
FLAGGED_NE = [
    "Arunachal Pradesh",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Sikkim",
]


# Indicators
@dataclass(frozen=True)
class Ind:
    key: str  # Short name
    match: str  # Phrase identifying NFHS column
    label: str  # Short label for charts
    unit: str
    better: int  # +1 if higher is better, -1 if lower is better
    group: str  # Outcome | Care | Cost and Cover | Context
    exclude: tuple = ()  # Phrases that should not appear


INDICATORS = [
    # Outcomes: Child Survival, Nutrition, Maternal Health
    Ind(
        "imr",
        "infant mortality rate",
        "Infant mortality",
        "deaths per 1,000 live births",
        -1,
        "Outcome",
    ),
    Ind(
        "nmr",
        "neonatal mortality rate",
        "Neonatal mortality",
        "deaths per 1,000 live births",
        -1,
        "Outcome",
    ),
    Ind(
        "u5mr",
        "under-five mortality rate",
        "Under-5 mortality",
        "deaths per 1,000 live births",
        -1,
        "Outcome",
    ),
    Ind(
        "stunt",
        "who are stunted",
        "Stunting",
        "% of children under 5",
        -1,
        "Outcome",
    ),
    Ind(
        "wast",
        "who are wasted",
        "Wasting",
        "% of children under 5",
        -1,
        "Outcome",
        ("severely",),
    ),
    Ind(
        "underwt",
        "who are underweight",
        "Underweight",
        "% of children under 5",
        -1,
        "Outcome",
    ),
    Ind(
        "anaem_child",
        "6-59 months who are anaemic",
        "Child anaemia",
        "% of children 6–59 months",
        -1,
        "Outcome",
    ),
    Ind(
        "anaem_preg",
        "pregnant women age 15-49",
        "Anaemia in pregnancy",
        "% of pregnant women",
        -1,
        "Outcome",
    ),
    Ind(
        "anaem_women",
        "all women age 15-49 years who are anaemic",
        "Anaemia, all women",
        "% of women 15–49",
        -1,
        "Outcome",
    ),
    Ind(
        "lowbmi",
        "body mass index",
        "Women with low BMI",
        "% of women 15–49",
        -1,
        "Outcome",
    ),
    # Care: Antenatal, Postnatal, Infant, Young Child
    Ind(
        "anc1",
        "first trimester",
        "ANC in 1st trimester",
        "% of mothers",
        1,
        "Care",
    ),
    Ind(
        "anc4",
        "at least 4 antenatal",
        "4+ antenatal visits",
        "% of mothers",
        1,
        "Care",
    ),
    Ind(
        "ifa",
        "iron folic acid",
        "Iron-folic acid 180+ days",
        "% of mothers",
        1,
        "Care",
    ),
    Ind(
        "inst",
        "institutional births",
        "Institutional births",
        "% of births",
        1,
        "Care",
    ),
    Ind(
        "pnc",
        "postnatal care from",
        "Postnatal care ≤2 days",
        "% of children",
        1,
        "Care",
    ),
    Ind(
        "vacc",
        "vaccination card or mother",
        "Full vaccination",
        "% of children 12–23 months",
        1,
        "Care",
    ),
    Ind(
        "vita",
        "vitamin a",
        "Vitamin A dose",
        "% of children 9–59 months",
        1,
        "Care",
    ),
    Ind(
        "ebf",
        "exclusively breastfed",
        "Exclusive breastfeeding",
        "% of infants under 6 months",
        1,
        "Care",
    ),
    Ind(
        "diet",
        "total children age 6-23 months receiving an adequate diet",
        "Adequate diet 6–23 m",
        "% of children",
        1,
        "Care",
    ),
    # Cost and Cover: Money reaching families
    Ind(
        "oop",
        "out-of-pocket",
        "Out-of-pocket cost per delivery",
        "₹ per delivery (public facility)",
        -1,
        "Cost & cover",
    ),
    Ind(
        "insur",
        "health insurance",
        "Health insurance cover",
        "% of households",
        1,
        "Cost & cover",
    ),
    # Context: Factors other than health spending
    Ind(
        "san",
        "improved sanitation",
        "Improved sanitation",
        "% of population",
        1,
        "Context",
    ),
    Ind(
        "water",
        "drinking-water",
        "Improved drinking water",
        "% of population",
        1,
        "Context",
    ),
    Ind(
        "school",
        "women with 10 or more years of schooling",
        "Women with 10+ years of schooling",
        "% of women 15–49",
        1,
        "Context",
    ),
    Ind(
        "lit",
        "women who are literate",
        "Women who are literate",
        "% of women 15–49",
        1,
        "Context",
    ),
]
IND = {i.key: i for i in INDICATORS}

OUTCOME_INDEX = [
    "imr",
    "u5mr",
    "stunt",
    "wast",
    "underwt",
    "anaem_child",
    "anaem_preg",
]
CARE_INDEX = ["anc1", "anc4", "ifa", "inst", "pnc", "vacc"]
CARE_PATHWAY = ["anc1", "anc4", "ifa", "inst", "pnc", "vacc"]


# Parsing
def _num(s: pd.Series) -> pd.Series:
    """Converts comma-separated figures to numbers and anything unreadable becomes NaN"""
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def _find(columns, ind: Ind):
    """Returns the first column whose name contains the indicator's phrase, else None"""
    for c in columns:
        name = str(c).lower()
        if ind.match in name and not any(x in name for x in ind.exclude):
            return c
    return None


def read_two_row_header(path: Path):
    """Reads a CSV whose header is 2 rows: Indicator and whether it is Urban/Rural/Total
    Returns the data rows and a list of (indicator, area) pairs, one per column.
    """
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    row1, row2 = raw.iloc[0].tolist(), raw.iloc[1].tolist()
    names, current = [], None
    for top, sub in zip(row1, row2):
        if top.strip():
            current = top
        names.append((current, sub.strip() or None))
    body = raw.iloc[2:].reset_index(drop=True)
    body = body[body[0].str.strip() != ""].reset_index(
        drop=True
    )  # Drop the blank rows
    return body, names


def to_long(body: pd.DataFrame, names) -> pd.DataFrame:
    """Reshape the DataFrame to one row per (State, Area), with the base columns repeated for each area."""
    base = {}
    per_area = {a: {} for a in AREAS}
    for i, (name, sub) in enumerate(names):
        if name is None:
            continue
        name = name.strip()
        if name in BASE_COLS and sub is None:
            base[name] = body[i]
        elif sub in AREAS:
            per_area[sub][name] = body[i]
    frames = []
    for a in AREAS:
        d = pd.DataFrame(base)
        for c in BASE_COLS[1:]:
            d[c] = _num(d[c])
        d["State"] = d["State"].str.strip()
        d["Area"] = a
        for name, col in per_area[a].items():
            d[name] = _num(col)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# Full NFHS names of indicator columns
FULL_NAMES: dict[str, str] = {}


def _short(name: str) -> str:
    """Strips trailing digits and '%' from NFHS column names"""
    n = re.sub(r"(?<=[a-z)])\d+(,\s*\d+)*", "", name.strip())
    n = re.sub(r"\s*\(%\)\s*$", "", n).strip()
    return n


def tidy(long: pd.DataFrame) -> pd.DataFrame:
    """Keeps State, Area and base columns and renames known indicators to their short keys.
    Other numeric columns are kept as 'x_<name>' to ensure that nothing in the data is lost
    """
    out = long[["State", "Area"] + BASE_COLS[1:]].copy()
    out["State"] = out["State"].str.strip().replace(STATE_FIXES)
    used = set()
    for ind in INDICATORS:
        c = _find(long.columns, ind)
        out[ind.key] = long[c].values if c is not None else np.nan
        if c is not None:
            used.add(c)
            FULL_NAMES[ind.key] = str(c).strip()
    for c in long.columns:
        if (
            c in used
            or c in BASE_COLS
            or c == "Area"
            or not pd.api.types.is_numeric_dtype(long[c])
        ):
            continue
        key = "x_" + re.sub(r"[^a-z0-9]+", "_", _short(str(c)).lower()).strip(
            "_"
        )
        out[key] = long[c].values
        FULL_NAMES[key] = str(c).strip()
    return out


def indicator_keys(df: pd.DataFrame) -> list[str]:
    """All indicator columns along with 'x_' columns"""
    return [c for c in df.columns if c in IND or c.startswith("x_")]


def label_of(key: str) -> str:
    """Short chart label for an indicator key."""
    return IND[key].label if key in IND else _short(FULL_NAMES.get(key, key))


# Raw file audit
def audit_raw(path: Path) -> dict:
    """Problems in the raw file fixed by data cleaning"""
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    body, names = read_two_row_header(path)
    h1 = [h for h in raw.iloc[0].tolist() if h.strip()]
    text_numbers = 0  # Comma-separated numbers
    for i, (name, sub) in enumerate(names):
        if name and name.strip() in BASE_COLS[1:]:
            text_numbers += int(body[i].str.contains(",").sum())
    empty_cols = int(
        sum(
            1
            for i in range(raw.shape[1])
            if (raw.iloc[:, i].str.strip() == "").all()
        )
    )
    messy_headers = sum(
        1
        for h in h1
        if h != h.strip() or re.search(r"[a-z)]\d", h) or "  " in h
    )
    states = body[0].str.strip()
    return {
        "text_numbers": text_numbers,
        "empty_cols": empty_cols,
        "messy_headers": messy_headers,
        "misspelt": sorted(s for s in states if s in STATE_FIXES),
        "n_states": int(states.nunique()),
        "n_indicators": len({n for n, sub in names if sub in AREAS}),
    }


# Loading the cleaned file
def load_clean_file(path: Path) -> pd.DataFrame:
    """Reads the cleaned file. Each indicator has three columns ending in ' (Urban)', ' (Rural)' and ' (Total)',
    and is reshaped to one row per (State, Area)"""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    base = [c for c in BASE_COLS if c in df.columns]
    frames = []
    for a in AREAS:
        suffix = f" ({a})"
        cols = {
            c: c[: -len(suffix)].strip()
            for c in df.columns
            if c.endswith(suffix) and c not in BASE_COLS
        }
        part = df[base].copy()
        part["Area"] = a
        part = pd.concat([part, df[list(cols)].rename(columns=cols)], axis=1)
        frames.append(part)
    return tidy(pd.concat(frames, ignore_index=True))


# Derived Variables
def minmax_score(s: pd.Series, better: int) -> pd.Series:
    """Rescales to 0-100: 100 represents the best state when higher is better (flipped when lower is better)"""
    rng = s.max() - s.min()
    z = (s - s.min()) / rng * 100 if rng else s * 0 + 50
    return z if better == 1 else 100 - z


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Adds region, spending per person, budget ratios, the two indices and spending tiers."""
    d = df.copy()
    d["Region"] = d["State"].map(REGION_OF).fillna("Other")
    pop = d["Population (Total)"]
    d["pc"] = d["Accounts"] * LAKH / pop  # Actual expenditure per person
    d["pc_be"] = (
        d["Budget Estimates"] * LAKH / pop
    )  # Budget estimate per person
    d["pc_re"] = (
        d["Revised Estimates"] * LAKH / pop
    )  # Revised estimate per person
    d["spend_cr"] = d["Accounts"] / 100  # Expenditure in ₹ crores
    d["pop_m"] = pop / 1e6  # Population in million
    d["util_be"] = (
        d["Accounts"] / d["Budget Estimates"] * 100
    )  # Actual as a percentage of budget estimate
    d["util_re"] = (
        d["Revised Estimates"] / d["Budget Estimates"] * 100
    )  # Revised estimate as a percentage of budget estimate
    d["flagged_ne"] = d["State"].isin(
        FLAGGED_NE
    )  # Flagged northeastern states
    for a in AREAS:
        m = d.Area == a
        for k in OUTCOME_INDEX + CARE_INDEX:
            d.loc[m, f"s_{k}"] = minmax_score(d.loc[m, k], IND[k].better)
        d.loc[m, "outcome_idx"] = d.loc[
            m, [f"s_{k}" for k in OUTCOME_INDEX]
        ].mean(axis=1)
        d.loc[m, "care_idx"] = d.loc[m, [f"s_{k}" for k in CARE_INDEX]].mean(
            axis=1
        )
        d.loc[m, "tier"] = pd.qcut(
            d.loc[m, "pc"],
            4,
            labels=["Lowest", "Lower-mid", "Upper-mid", "Highest"],
        ).astype(str)
    return d


# Statistics helper functions
def spearman(x, y):
    """Spearman correlation with p-value and 95% confidence interval"""
    m = pd.notna(x) & pd.notna(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    n = len(x)
    if n < 5:
        return dict(r=np.nan, p=np.nan, n=n, lo=np.nan, hi=np.nan)
    r, p = stats.spearmanr(x, y)
    se = np.sqrt((1 + r * r / 2) / (n - 3))
    z = np.arctanh(np.clip(r, -0.999, 0.999))
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return dict(r=float(r), p=float(p), n=int(n), lo=float(lo), hi=float(hi))


def partial_spearman(x, y, z):
    """Spearman correlation of x and y after removing the effect of z"""
    m = pd.notna(x) & pd.notna(y) & pd.notna(z)
    rx, ry, rz = (stats.rankdata(np.asarray(v)[m]) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r = float(np.corrcoef(ex, ey)[0, 1])
    n = int(m.sum())
    t = r * np.sqrt((n - 3) / max(1e-12, 1 - r * r))
    return dict(r=r, p=float(2 * stats.t.sf(abs(t), n - 3)), n=n)


def log_fit(x, y):
    """OLS of y on log10(x): returns slope and intercept for trend lines"""
    m = pd.notna(x) & pd.notna(y) & (np.asarray(x) > 0)
    b, a = np.polyfit(np.log10(np.asarray(x)[m]), np.asarray(y)[m], 1)
    return b, a


def fmt_p(p):
    """Formats p-values for captions"""
    if p is None or np.isnan(p):
        return "p = n/a"
    return (
        "p < 0.001"
        if p < 0.001
        else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"
    )


def inr(v, dec=0):
    """Grouping of digits with commas as per Indian System"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    neg = v < 0
    s = f"{abs(v):.{dec}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        head = (
            ",".join(re.findall(r"\d{1,2}", head[::-1]))[::-1] if head else ""
        )
        whole = head + "," + tail
    out = whole + ("." + frac if frac else "")
    return ("−" if neg else "") + "₹" + out


# Outlier Detection
def _grubbs_flags(s: pd.Series, alpha=0.05) -> list:
    """Two sided Grubbs' Test: most extreme value is removed while it is significant
    and index labels (state names) of flagged values are returned"""
    s = s.dropna().copy()
    flagged = []
    while len(s) >= 7:
        n = len(s)
        dev = (s - s.mean()).abs()
        g = dev.max() / s.std(ddof=1)
        t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        crit = (n - 1) / np.sqrt(n) * np.sqrt(t * t / (n - 2 + t * t))
        if g <= crit:
            break
        idx = dev.idxmax()
        flagged.append(idx)
        s = s.drop(idx)
    return flagged


def outlier_grid(raw: pd.DataFrame) -> pd.DataFrame:
    """Runs the five outlier detection methods on raw data and counts state-level flags
    using the same preprocessing and parameter settings as the data cleaning notebook
    """
    keys = indicator_keys(raw)
    wide = raw.pivot_table(
        index="State", columns="Area", values=keys, dropna=False, sort=False
    )
    wide.columns = [f"{k} ({a})" for k, a in wide.columns]
    base = raw[raw.Area == "Total"].set_index("State")[
        BASE_COLS[1:] + ["pc", "pc_be", "pc_re"]
    ]
    X = pd.concat([base, wide], axis=1)
    X = X.loc[:, X.notna().any()]  # Drop columns without any data
    res = pd.DataFrame(
        0,
        index=X.index,
        columns=["IQR", "Modified z", "Grubbs", "DBSCAN", "LOF"],
    )
    # Univariate Methods
    for c in X.columns:
        s = X[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        res.loc[
            s.index[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)], "IQR"
        ] += 1
        mad = (s - s.median()).abs().median()
        if mad:
            res.loc[
                s.index[(0.6745 * (s - s.median()) / mad).abs() > 3.5],
                "Modified z",
            ] += 1
        for state in _grubbs_flags(s):
            res.loc[state, "Grubbs"] += 1
    # Multivariate Methods (1=flagged, 0=not flagged)
    Z = StandardScaler().fit_transform(
        SimpleImputer(strategy="mean").fit_transform(X)
    )
    P = PCA(n_components=0.90, random_state=42).fit_transform(Z)
    res["DBSCAN"] = (
        DBSCAN(eps=12.0, min_samples=3).fit_predict(P) == -1
    ).astype(int)
    res["LOF"] = (
        LocalOutlierFactor(n_neighbors=3).fit_predict(P) == -1
    ).astype(int)
    res.attrs["n_cols"] = X.shape[1]
    res.attrs["n_components"] = P.shape[1]
    return res


# Loading everything
@dataclass
class Bundle:
    """Everything required by the charts is loaded once the app starts"""

    raw: (
        pd.DataFrame
    )  # Tidied raw file with derived variables but no imputation
    clean: (
        pd.DataFrame
    )  # Cleaned file with imputation, derived variables and all three areas
    total: (
        pd.DataFrame
    )  # Clean rows where Area == "Total", with one row per state
    imputed: (
        pd.DataFrame
    )  # True where a cell was missing in raw and filled in clean
    audit: dict  # Problems found in the raw file
    outliers: pd.DataFrame  # Outlier flags per state and method
    geojson: dict  # State boundaries for the map
    source: str  # Name of the cleaned file used


def load_all(
    raw_path: Path = RAW_PATH, clean_path: Path = CLEAN_PATH
) -> Bundle:
    """Loads the raw and cleaned files and builds everything needed by the charts"""
    if not Path(clean_path).exists():
        raise FileNotFoundError(
            f"Cleaned file not found: {clean_path}. Run the cleaning notebook first."
        )
    body, names = read_two_row_header(raw_path)
    raw = tidy(to_long(body, names))
    cleaned = load_clean_file(clean_path)
    missing_areas = [a for a in AREAS if a not in set(cleaned.Area)]
    if missing_areas:
        raise ValueError(
            f"Cleaned file has no rows for: {', '.join(missing_areas)}"
        )
    keys = indicator_keys(raw)
    for k in keys:
        if k not in cleaned.columns:
            cleaned[k] = np.nan
    # Same row order as the raw file for cell-by-cell comparison
    cleaned = (
        cleaned.set_index(["State", "Area"])
        .reindex(raw.set_index(["State", "Area"]).index)
        .reset_index()
    )
    cleaned[keys] = cleaned[keys].round(2)
    imputed = raw[keys].isna() & cleaned[keys].notna()
    imputed.insert(0, "Area", raw["Area"].values)
    imputed.insert(0, "State", raw["State"].values)
    raw_d = add_derived(raw)
    clean_d = add_derived(cleaned)
    total = clean_d[clean_d.Area == "Total"].reset_index(drop=True)
    with open(GEOJSON_PATH) as f:
        gj = json.load(f)
    return Bundle(
        raw=raw_d,
        clean=clean_d,
        total=total,
        imputed=imputed,
        audit=audit_raw(raw_path),
        outliers=outlier_grid(raw_d),
        geojson=gj,
        source=Path(clean_path).name,
    )


if __name__ == "__main__":
    # Test with python data.py
    b = load_all()
    t = b.total
    print(f"Data source: {b.source}")
    print(f"States: {len(t)} | indicators: {len(indicator_keys(b.raw))}")
    print(
        f"Spending per person: {inr(t.pc.min())} to {inr(t.pc.max())} (median {inr(t.pc.median())})"
    )
    for key, name in [
        ("outcome_idx", "Outcome Index"),
        ("care_idx", "Care Index"),
        ("imr", "Infant mortality"),
    ]:
        s = spearman(t.pc, t[key])
        print(f"Spending vs {name}: rho = {s['r']:+.2f}, {fmt_p(s['p'])}")
    print(
        "Imputed cells by area:",
        b.imputed.groupby("Area")[indicator_keys(b.raw)]
        .sum()
        .sum(axis=1)
        .to_dict(),
    )
    o = b.outliers
    print(
        f"Outliers ({o.attrs['n_cols']} columns, {o.attrs['n_components']} PCA components): "
        f"DBSCAN {o.index[o.DBSCAN == 1].tolist()}, LOF {o.index[o.LOF == 1].tolist()}"
    )
