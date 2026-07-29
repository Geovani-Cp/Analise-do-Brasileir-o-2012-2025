# AGENTS.md – Quick reference for OpenCode agents

## Repository layout
- `Untitled.ipynb` – main notebook that loads the CSV from https://www.football-data.co.uk/new/BRA.csv and performs exploratory analysis / feature engineering.
- `.ipynb_checkpoints/` – Jupyter checkpoint files, can be ignored.
- `db/project_filebrowser.db` – internal Jupyter file‑browser DB, not relevant to analysis.

## Common pitfalls an agent might miss
- **Directory name contains a space**: the root folder is `odds_brasileirão ` (note the trailing space before the slash). All file paths must include that space exactly, e.g. `"/home/geovani/Documentos/Projetos/odds_brasileirão /anaconda_projects/Untitled.ipynb"`.
- **Data source is remote**: the notebook reads the data with  
  ```python
  df = pd.read_csv("https://www.football-data.co.uk/new/BRA.csv")
  ```  
  An offline environment will fail; internet access is required.
- **Column pruning and NaN removal**: after loading, the notebook keeps only `['Home','Away','HG','AG','Res','PSCH','PSCD','PSCA']` and then calls `df = df.dropna()`. This silently discards all rows missing any of the listed odds (most rows for B365, BFE, etc.). Keep a copy of the original `df` if you need those columns later.
- **Date/Time are strings**: `Date` is `dd/mm/yyyy` and `Time` is `HH:MM`. They are not parsed in the original notebook, so any temporal feature (rolling stats, season split) must first convert them:
  ```python
  df['date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
  df['time'] = pd.to_timedelta(df['Time'] + ':00')
  df['datetime'] = df['date'] + df['time']
  ```
- **Rolling / EWMA calculations require chronological order**. When creating lag or exponential‑weighted features, sort by `datetime` first, otherwise the weights are applied to a random order.

## Recommended commands for a quick iteration
```bash
# From the project root (mind the space in the folder name)
cd "/home/geovani/Documentos/Projetos/odds_brasileirão /anaconda_projects"

# Launch Jupyter Lab / Notebook
jupyter lab           # or: jupyter notebook
```
*If the environment lacks required packages:*
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Typical workflow an agent should follow
1. **Start Jupyter** and open `Untitled.ipynb`.
2. **Run the first cell** to import `pandas` and `numpy`.
3. **Run the CSV loading cell** – verify that the download succeeds.
4. **Inspect `df.info()`** to see missing‑value pattern.
5. **If you need additional columns**, comment‑out the column‑selection line (`df = df.loc[:, [...]]`) and keep the full DataFrame.
6. **Create temporal columns** (`date`, `time`, `datetime`) before any rolling/lag feature engineering.
7. **When adding EWMA or rolling stats**, always:
   ```python
   df = df.sort_values('datetime')
   df['home_ewma'] = df.groupby('Home')['HG'].transform(
       lambda s: s.shift().ewm(span=5, adjust=False).mean())
   ```
8. **Save any intermediate DataFrames** only to `/tmp/opencode/` (temporary folder) to avoid touching the repo files.

## Gotchas for testing / reproducibility
- The notebook does not contain any unit tests; any verification must be done manually (e.g., `df.head()`, `df.describe()`).
- Because the data URL changes each season, pinning a specific CSV version is recommended for reproducible runs (download once, store locally, and adjust the path).

## References inside the repo
- The notebook itself is the single source of truth for data handling and feature engineering.
- No CI, lint, or type‑check configuration exists; agents can ignore those steps.

---
*Keep this file short and focused; update only when you add new scripts, change the data source, or introduce a build environment.*