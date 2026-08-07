# AGENTS.md – Quick reference for OpenCode agents

## IMPORTANT – path handling
- The repository root ends with a space: `odds_brasileirão `. **All absolute paths must include that trailing space** (e.g. `"/home/geovani/Documentos/Projetos/odds_brasileirão /anaconda_projects/..."`).

## Data pipeline
```
preprocessamento_brasileirao.ipynb  →  dados_processados.csv  →  LogisticRegression.ipynb / RandomForest.ipynb / analise_descristiva.ipynb
```

## Execution order
1. Run **all cells** of `preprocessamento_brasileirao.ipynb` to regenerate `dados_processados.csv` **before** opening the other notebooks.
2. `analise_descristiva.ipynb` reads the CSV for EDA.
3. `LogisticRegression.ipynb` and `RandomForest.ipynb` both read the CSV and add extra features inline.

## Kernel
- Use the **`map_pass`** kernel (Python 3.12). If unavailable, create any Python 3 kernel.
- Missing libs: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels`

## Regeneration commands
```bash
# Full pipeline (in order)
jupyter nbconvert --to notebook --execute preprocessamento_brasileirao.ipynb --output preprocessamento_brasileirao.ipynb
jupyter nbconvert --to notebook --execute LogisticRegression.ipynb --output LogisticRegression.ipynb
jupyter nbconvert --to notebook --execute RandomForest.ipynb --output RandomForest.ipynb
```
- For quick ad-hoc experiments, write temp CSVs to `/tmp/opencode/` — never commit them.

## Data source
- Raw: `https://www.football-data.co.uk/new/BRA.csv` (Brazilian Serie A, 2012–2025)
- 5525 raw matches → **5275** after dropping rows with missing odds (`PSCH`/`PSCD`/`PSCA`).
- `dados_processados.csv` is **git-ignored**; created fresh each run.

## Generated CSV profile
- **33 columns**, 5275 rows (after `dropna()`).
- Columns: `Date`, `Season`, `Home`, `Away`, `HG`, `AG`, `Res`, `PSCH`, `PSCD`, `PSCA`, `Date_parsed`, `p_home`, `p_draw`, `p_away`, `dif_prob`, `dif_draw`, `goal_diff`, `goal_total`, `bucket_favoritismo`, `home_last5_goals`, `away_last5_goals`, `home_last5_concede`, `away_last5_concede`, `home_ewma_5_goals`, `away_ewma_5_goals`, `Res_num`, `home_last5_winrate`, `away_last5_winrate`, `home_points`, `away_points`, `home_last5_ppg`, `away_last5_ppg`, `entropy`.
- **`Time` column is dropped** — only `Date` is kept (string `dd/mm/YYYY`). For intra-day ordering, keep `Time` from the raw source.

## Feature-engineering (in `preprocessamento_brasileirao.ipynb`)
- **Odds → implied probabilities** (`p_home`, `p_draw`, `p_away`), juice-removed `dif_prob`.
- **Form metrics** (last-5 rolling): goals, conceded, win-rate, points-per-game, EWMA of goals.
- **Entropy** of implied probabilities (normalized 0–1).
- **Categorical bucket** (`bucket_favoritismo`): 7 quantile bins of `dif_prob`.

## Key gotchas you might miss
- **Chronology** — all rolling/EWMA features use `.shift()` *after* sorting by `Date_parsed`. If you add new temporal features, **re-sort before every rolling call**. Every `.shift()` is shift=1 (leakage prevention).
- **First-5-row NaNs** — rolling features are `NaN` for the first 5 matches of a team; EWMA stays `NaN` until its warm-up period.
- **`home_last5_ppg` bug (FIXED)** — current code correctly groups by `"Home"` on `"home_points"` (cell 88). Old bug was `groupby("Home")["away_points"]`.
- **Leakage prevention** — always `.shift()` before any aggregation using past match data. `bucket_favoritismo` is safe (pre-match odds only).
- **Split must be temporal** — both modeling notebooks use chronological 80/20 split (`sort_values('Date_parsed')` + `iloc[:cut]`). Never use `train_test_split` with `random_state`.

## Modeling notebooks — what exists

### `LogisticRegression.ipynb`
- **Model**: `LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')`.
- **Feature set** (6 features): `dif_prob`, `diff_winrate`, `ratio_concede_capped`, `market_form_divergence`, `diferenca_ajustada`, `entropy_squared`.
- **Scaling**: `StandardScaler` — but has a **leakage bug**: calls `fit_transform` on both `X_train` and `X_test` separately (should `transform` the test set).
- **Metrics**: accuracy, classification report, AUC-ROC, Brier, LogLoss.
- **Calibration**: `CalibratedClassifierCV`.

### `RandomForest.ipynb`
- **Model**: `RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)`.
- **Feature set** (17 features): `dif_prob`, `entropy`, `p_home`, `p_draw`, `p_away`, `diff_goals`, `diff_winrate`, `diff_ppg`, `home_goal_ratio`, `away_goal_ratio`, `ratio_concede_capped`, `market_form_divergence`, `market_form_direction`, `log_diff_goal_ratio`, `diferenca_ajustada`, `home_ewma_5_goals`, `away_ewma_5_goals`, `dif_draw`.
- **No scaling** (tree-based, not needed).
- **Metrics**: accuracy, classification report, Brier, LogLoss, AUC-ROC, feature importances.
- **NaN/Inf handling**: `X.replace([np.inf, -np.inf], np.nan).fillna(0)`.
- **Baseline**: `DummyClassifier` accuracy ~0.52. RF achieves ~0.59.

### `analise_descristiva.ipynb`
- Pure EDA (visualizations, distributions, correlations). Reads only `dados_processados.csv`. No modeling.

## Target
- **Binary**: `target_bin = (Res == 'H').astype(int)` (home win vs not).
- Multi-class also defined: `target = Res.map({'H': 0, 'D': 1, 'A': 2})`.

## Features created in the modeling notebooks (post-processing)
- **Relative differences**: `diff_goals`, `diff_winrate`, `diff_ppg`.
- **Efficiency ratios**: `home_goal_ratio`, `away_goal_ratio`.
- **Capped features**: `diff_goal_ratio_capped`, `ratio_concede_capped` (percentiles 1/99).
- **Log-transformed**: `log_diff_goal_ratio`.
- **Market vs form interaction**: `market_form_divergence` (absolute diff), `market_form_direction` (signed diff).
- **Other**: `diferenca_ajustada`, `entropy_squared`, `dif_draw`.

## Files summary
| File | Purpose |
|------|---------|
| `preprocessamento_brasileirao.ipynb` | ETL + feature engineering → `dados_processados.csv` |
| `LogisticRegression.ipynb` | Logistic regression pipeline (6 features, StandardScaler — has leakage bug) |
| `RandomForest.ipynb` | Random Forest pipeline (17 features, richer feature set) |
| `analise_descristiva.ipynb` | EDA/visualization |
| `dados_processados.csv` | Generated artifact (git-ignored) |

---
*Keep this file short. Update when you add/rename notebooks, change the data source, or discover new pipeline quirks.*