# AGENTS.md – Quick reference for OpenCode agents

## IMPORTANT – path handling
- The repository root ends with a space: `odds_brasileirão `. **All absolute paths must include that trailing space** (e.g. `"/home/geovani/Documentos/Projetos/odds_brasileirão /anaconda_projects/..."`).

## Running the notebooks
- Use the **`map_pass`** kernel (Python 3.10). If unavailable, create any Python 3 kernel after installing the required libs:
  ```bash
  pip install pandas numpy matplotlib seaborn
  ```
- Execution order matters: run **all cells** of `Untitled.ipynb` to (re)generate `dados_processados.csv` before opening `analise_descristiva.ipynb`.

## Data pipeline (primary flow)
```
Untitled.ipynb  →  dados_processados.csv  →  analise_descristiva.ipynb
```
- `dados_processados.csv` is **git‑ignored**; it is created fresh each run.
- The CSV contains 35 columns (5275 rows after `dropna`).

## Key gotchas you might miss
- **Chronology** – rolling / EWMA features use `.shift()` *after* sorting by `Date_parsed`. If you add new temporal features, **re‑sort before every rolling call**.
- **First‑5‑row NaNs** – rolling features are `NaN` for the first 5 matches of a team; EWMA stays `NaN` until its warm‑up period.
- **Bug in `home_last5_ppg`** – the notebook mistakenly groups by `Home` while aggregating `away_points`. Fix by aggregating `home_points`.
- **Time column omitted** – only `Date` is kept. For intra‑day ordering you must keep `Time` and build a `datetime` column.
- **Leakage prevention** – always `.shift()` before any aggregation that uses past match data. `bucket_favoritismo` is safe because it uses only pre‑match odds.

## Minimal command cheat‑sheet
```bash
# Regenerate processed data
jupyter nbconvert --to notebook --execute Untitled.ipynb --output Untitled.ipynb

# Quick visual inspection (run after regeneration)
jupyter notebook analise_descristiva.ipynb
```
- For ad‑hoc experiments, write temporary CSVs to `/tmp/opencode/` – never modify the repo files.

## Feature‑engineering summary (what exists)
- **Odds** → implied probabilities, juice‑removed `dif_prob`.
- **Form metrics** (last‑5 rolling): goals, conceded, win‑rate, points‑per‑game, EWMA of goals.
- **Entropy** of implied probabilities.
- **Return** columns (`ret_home`, etc.) for betting‑return analysis.
- **Categorical bucket** (`bucket_favoritismo`) based on `dif_prob` quantiles.

## Modelling tips (if you add a model notebook)
- **Target**: `Res` → map `'H':0, 'D':1, 'A':2` or binary `target_bin = (Res=='H')` for a simpler problem.
- **Balance**: use `class_weight='balanced'` (RandomForest) or SMOTE if you prefer oversampling.
- **Relative features** are often more predictive than absolute ones:
  - `diff_goals = home_last5_goals - away_last5_goals`
  - `diff_goal_ratio = home_goal_ratio - away_goal_ratio`
  - `market_vs_form = dif_prob / (diff_winrate+1e-6)`
- **Temporal validation** – split chronologically (first 80 % of dates for training, last 20 % for testing) to avoid look‑ahead bias.

---
*Keep this file short. Update only when you add new notebooks, change the data source, or discover new pipeline quirks.*
