# Health Spending and Maternal and Child Health in India

**Question:** Do states that spend more per capita on health have better maternal and child health outcomes?

A Plotly Dash dashboard comparing state health spending (RBI, 2019-20) with NFHS-5 (2019-21) indicators for 29 states.

The live static version can be found here: https://iitm-da25b042.github.io/DCV-Project-2026/
(Hover and zoom work. However, the interactive state selector and chart controls require the Dash app.)

## Run the dashboard
```bash
pip install -r requirements.txt
python app.py        # Open http://127.0.0.1:8050, stop with Ctrl+C
```

## Contents

| Path | Description |
|---|---|
| `data/raw/` | Original downloaded files: RBI State Finances (Excel), NFHS-5 state indicators (CSV), NHM population projections (PDF) |
| `data/interim/budget_nfhs_combined.csv` | Spending, population and NFHS-5 indicators combined into one csv file, before cleaning |
| `data/clean/budget_nfhs_combined_clean.csv` | Cleaned file used by the dashboard |
| `data_cleaning.ipynb` | Data cleaning on interim file -> cleaned file |
| `data_visualisation.ipynb` | All dashboard charts with their captions |
| `app.py` | Dash app with 6 tabs, state highlight and chart controls |
| `figures.py` | One function per chart, each returning the figure and its caption |
| `data.py` | Loads the data and computes spending per capita, indices and statistics |
| `theme.py`, `assets/style.css` | Chart and page styling |
| `export_static.py`, `docs/index.html` | Static copy of the dashboard, published with GitHub Pages |
| `data/geo/`, `tools/` | State map file and the script used to build it |
| `screenshots/` | Screenshots of dashboard charts |
| `report.pdf` | Project report |
| `data_dictionary.pdf` | Column names, types, units, sources and allowed ranges of the cleaned data |
| `cleaning_log.pdf` | Cleaning steps and their outcomes |
| `contribution.pdf` | Contributions of each team member |
| `ai_use.pdf` | Use of AI tools |

## Data sources

- Reserve Bank of India, *State Finances: A Study of Budgets*: revenue expenditure on Medical and Public Health, 2019-20 (Accounts, Budget Estimates, Revised Estimates)
- National Family Health Survey (NFHS-5), 2019-21: state indicators (Total, Urban, Rural)
- National Commission on Population, *Population Projections for India and States 2011-2036*: projected population, 1 March 2020

All files were downloaded manually from the publishers' websites and no scraping was used.

Methods, findings and limitations can be found in Tab 6 of the dashboard.