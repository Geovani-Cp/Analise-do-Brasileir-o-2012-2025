# AGENTS.md – Quick reference for OpenCode agents

## IMPORTANT – path handling
- The repository root ends with a space: `odds_brasileirão `. **All absolute paths must include that trailing space** (e.g. `"/home/geovani/Documentos/Projetos/odds_brasileirão /anaconda_projects/..."`).

## Running the notebooks
- Use the **`map_pass`** kernel (Python 3.12). If unavailable, create any Python 3 kernel after installing the required libs:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost
  ```
- Execution order matters: run **all cells** of `preprocessamento_brasileirao.ipynb` to (re)generate `dados_processados.csv` before opening `analise_descristiva.ipynb` or `modelos_prever.ipynb`.

## Data pipeline (primary flow)
```
preprocessamento_brasileirao.ipynb  →  dados_processados.csv  →  analise_descristiva.ipynb / modelos_prever.ipynb
```
- `dados_processados.csv` is **git-ignored**; it is created fresh each run.
- The CSV contains 35 columns (5275 rows after `dropna`).

## Key gotchas you might miss
- **Chronology** – rolling / EWMA features use `.shift()` *after* sorting by `Date_parsed`. If you add new temporal features, **re-sort before every rolling call**.
- **First-5-row NaNs** – rolling features are `NaN` for the first 5 matches of a team; EWMA stays `NaN` until its warm-up period.
- **Bug in `home_last5_ppg`** – the original notebook mistakenly grouped by `Home` while aggregating `away_points`. **FIXED in current code** (cell 67 uses `home_points`).
- **Time column omitted** – only `Date` is kept. For intra-day ordering you must keep `Time` and build a `datetime` column.
- **Leakage prevention** – always `.shift()` before any aggregation that uses past match data. `bucket_favoritismo` is safe because it uses only pre-match odds.
- **Split must be temporal** – `modelos_prever.ipynb` uses chronological 80/20 split (not random). Never use `train_test_split` with `random_state`.

## Minimal command cheat-sheet
```bash
# Regenerate processed data
jupyter nbconvert --to notebook --execute preprocessamento_brasileirao.ipynb --output preprocessamento_brasileirao.ipynb

# Quick visual inspection (run after regeneration)
jupyter notebook analise_descristiva.ipynb
```
- For ad-hoc experiments, write temporary CSVs to `/tmp/opencode/` – never modify the repo files.

## Feature-engineering summary (what exists in `preprocessamento_brasileirao.ipynb`)
- **Odds** to implied probabilities (`p_home`, `p_draw`, `p_away`), juice-removed `dif_prob`.
- **Form metrics** (last-5 rolling): goals, conceded, win-rate, points-per-game, EWMA of goals.
- **Entropy** of implied probabilities (normalized 0-1).
- **Categorical bucket** (`bucket_favoritismo`) based on `dif_prob` quantiles (7 bins).

## Features created in `modelos_prever.ipynb` (post-processing)
- **Relative differences**: `diff_goals`, `diff_winrate`, `diff_ppg`
- **Efficiency ratios**: `home_goal_ratio`, `away_goal_ratio`
- **Capped features**: `diff_goal_ratio_capped`, `ratio_concede_capped` (percentiles 1/99)
- **Log-transformed**: `log_diff_goal_ratio`
- **Market vs form interaction**: `market_form_divergence` (absolute difference), `market_form_direction` (signed difference)
- **Raw odds**: `PSCH`, `PSCD`, `PSCA` (kept for model)

## Final feature list used for modeling (`FEATURES_FINAIS` - 14 features)
```python
FEATURES_FINAIS = [
    # Mercado (4)
    'dif_prob', 'entropy', 'PSCH', 'PSCD', 'PSCA',
    # Forma relativa - diferencas (3)
    'diff_goals', 'diff_winrate', 'diff_ppg',
    # Eficiencia ofensiva (2)
    'home_goal_ratio', 'away_goal_ratio',
    # Vulnerabilidade relativa estabilizada (1)
    'ratio_concede_capped',
    # Interacao mercado-forma (2)
    'market_form_divergence', 'market_form_direction',
    # Eficiencia relativa em log (1)
    'log_diff_goal_ratio',
    # EWMA goals (2)
    'home_ewma_5_goals', 'away_ewma_5_goals',
]
```

## Modelling tips (current state in `modelos_prever.ipynb`)
- **Target**: `target_bin = (Res=='H')` for binary classification (home win vs not).
- **Baseline**: `DummyClassifier` accuracy ~0.52.
- **Random Forest**: `n_estimators=100, max_depth=10, class_weight='balanced'` -> ~0.59 accuracy, LogLoss ~0.66, Brier ~0.23.
- **Temporal validation** - split chronologically (first 80% of dates for training, last 20% for testing).
- **Next steps**: try XGBoost/LightGBM, add `TimeSeriesSplit` walk-forward CV, calibrate probabilities (Platt/Isotonic).

## Data source
- Raw data: `https://www.football-data.co.uk/new/BRA.csv` (Brazilian Serie A, 2012-2025)
- 5525 matches raw -> 5275 after dropping rows with missing odds (PSCH/PSCD/PSCA)

## Files in repo
| File | Purpose |
|------|---------|
| `preprocessamento_brasileirao.ipynb` | Main ETL + feature engineering -> generates `dados_processados.csv` |
| `modelos_prever.ipynb` | Modeling pipeline: feature selection, temporal split, RF baseline |
| `analise_descristiva.ipynb` | Exploratory data analysis / visualization |
| `dados_processados.csv` | Generated artifact (git-ignored) |
| `GUIA_MELHORIAS_FEATURES.md` | Detailed didactic guide for fixes & improvements (read this!) |

---
*Keep this file short. Update only when you add new notebooks, change the data source, or discover new pipeline quirks.*