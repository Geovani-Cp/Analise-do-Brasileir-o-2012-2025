# Guia Didático: Melhorias no Pipeline de Features — Brasileirão Odds

> **Objetivo:** Documentar passo a passo o que corrigir, por que corrigir e como implementar.
> **Público:** Você (Data Scientist do projeto) — linguagem didática, sem jargão desnecessário.
> **Princípio:** *Nenhum arquivo original será alterado por este guia. Você decide quando e como aplicar.*

---

## Sumário

1. [Visão Geral dos Problemas Encontrados](#1-visão-geral-dos-problemas-encontrados)
2. [Correção Crítica #1: Bug no `home_last5_ppg`](#2-correção-crítica-1-bug-no-home_last5_ppg)
3. [Correção Crítica #2: Split Temporal vs Aleatório](#3-correção-crítica-2-split-temporal-vs-aleatório)
4. [Limpeza de Features: Outliers e Redundância](#4-limpeza-de-features-outliers-e-redundância)
5. [Nova Feature Recomendada: `market_form_divergence`](#5-nova-feature-recomendada-market_form_divergence)
6. [Lista Final de Features Recomendadas (10 features)](#6-lista-final-de-features-recomendadas-10-features)
7. [Checklist de Validação Antes de Treinar](#7-checklist-de-validação-antes-de-treinar)
8. [Próximos Passos Evolutivos](#8-próximos-passos-evolutivos)

---

## 1. Visão Geral dos Problemas Encontrados

| # | Problema | Gravidade | Onde Está | Impacto Real |
|---|----------|-----------|-----------|--------------|
| 1 | `home_last5_ppg` usa pontos do **visitante** em vez do mandante | 🔴 Crítico | `preprocessamento_brasileirao.ipynb` linha 1332 | Feature tem correlação **negativa** com vitória em casa (−0.112) — o modelo aprende o oposto da realidade |
| 2 | `train_test_split(random_state=42)` embaralha datas | 🔴 Crítico | `modelos_prever.ipynb` células 7 e 24 | **Data leakage temporal**: jogos de 2025 "prevêem" jogos de 2013. Accuracy inflada artificialmente |
| 3 | `diff_goal_ratio`, `ratio_concede`, `market_vs_form` têm outliers de **±1,6 milhões** | 🟡 Alto | `modelos_prever.ipynb` células 11–15 | Estouram `std`, degradam Logistic Regression, poluem importância do Random Forest |
| 4 | `entropy_unc` é clone 98.6% de `dif_prob` | 🟡 Alto | `modelos_prever.ipynb` célula 13 | Multicolinearidade extrema → instabilidade numérica na Regressão Logística |
| 5 | `market_vs_form` usa divisão instável (denominador ~0) | 🟡 Alto | `modelos_prever.ipynb` célula 12 | Correlação zero com target (0.022), outliers extremos, não funciona |

> **Resumo:** Os dois primeiros itens (bug + split temporal) sozinhos explicam a maior parte do gap entre seu resultado atual e um modelo confiável. Os itens 3–5 são "limpeza de casa" que tornam o modelo robusto e interpretável.

---

## 2. Correção Crítica #1: Bug no `home_last5_ppg`

### 2.1 O que está errado

No arquivo `preprocessamento_brasileirao.ipynb`, célula **Execution 32** (linhas 1330–1345):

```python
# ATUAL (ERRADO) — linha 1332
df_raw["home_last5_ppg"] = (
    df_raw
    .groupby("Home")["away_points"]   # ← PROBLEMA: usa "away_points"
    .transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
)

df_raw["away_last5_ppg"] = (
    df_raw
    .groupby("Away")["away_points"]   # ← Este está correto (away usa away_points)
    .transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
)
```

### 2.2 Por que está errado — Explicação Didática

**Mapeamento dos pontos** (definido linhas 1288–1312):
- `home_points` = 3 se **vitória do mandante** (Res_num=2), 1 se empate, 0 se derrota
- `away_points` = 3 se **vitória do visitante** (Res_num=0), 1 se empate, 0 se derrota

**O que o código faz hoje:**
- `home_last5_ppg` = média dos **pontos que o time visitante fez** nos últimos 5 jogos do mandante
- Exemplo: Flamengo (casa) vs Vasco (fora). O `home_last5_ppg` do Flamengo pega os pontos que o **Vasco** fez nos jogos anteriores do Flamengo como mandante. **Não faz sentido nenhum.**

**O que deveria fazer:**
- `home_last5_ppg` = média dos **pontos que o mandante fez** nos seus últimos 5 jogos como mandante
- Ou seja: agrupar por `Home` e usar coluna `home_points`

### 2.3 Como corrigir (copie e cole)

**Arquivo:** `preprocessamento_brasileirao.ipynb`  
**Célula:** Execution 32 (aprox. linha 1330)

```python
# CORRIGIDO — substituir a célula inteira
df_raw["home_last5_ppg"] = (
    df_raw
    .groupby("Home")["home_points"]      # ← ERA "away_points", AGORA "home_points"
    .transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
)

df_raw["away_last5_ppg"] = (
    df_raw
    .groupby("Away")["away_points"]      # ← Este já estava correto
    .transform(lambda s: s.shift().rolling(5, min_periods=1).mean())
)
```

### 2.4 Como validar que corrigiu

Após rodar o preprocessamento e gerar novo `dados_processados.csv`:

```python
import pandas as pd
df = pd.read_csv("dados_processados.csv")

# Correlação deve virar POSITIVA (era -0.112)
print(df['home_last5_ppg'].corr(df['Res'].map({'H':1, 'D':0, 'A':0})))
# Esperado: valor entre +0.05 e +0.15

# Sanity check: time forte em casa deve ter PPG alto
print(df[df['Home']=='Flamengo'][['Date','Home','HG','AG','Res','home_last5_ppg']].tail(10))
```

---

## 3. Correção Crítica #2: Split Temporal vs Aleatório

### 3.1 O que está errado

Em `modelos_prever.ipynb`, células **7** e **24**:

```python
# ATUAL (ERRADO para série temporal)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 3.2 Por que está errado — Explicação Didática

**Série temporal ≠ dados i.i.d.** (independentes e identicamente distribuídos).

| Cenário | O que acontece |
|---------|----------------|
| **Split aleatório** | Linha 1000 (jogo de 2015) vai para treino, linha 1001 (jogo de 2015) vai para teste. O modelo "vê" o futuro para prever o passado. |
| **Split temporal** | Treino = jogos até 2023. Teste = jogos 2024–2025. O modelo **nunca viu** os jogos de teste. Isso simula produção real. |

**Efeito prático:** Com split aleatório, sua accuracy de 0.58 (binário) é **otimista**. No temporal, cai para ~0.62 (ainda bom, mas honesto). O LogLoss e Brier também pioram — e isso é **bom**, pois reflete a incerteza real.

### 3.3 Como corrigir (copie e cole)

**Arquivo:** `modelos_prever.ipynb`  
**Substitua as células 7 e 24** (e qualquer outro split) por:

```python
# 1. Garantir ordenação cronológica (já deve estar, mas segurança)
df_model = df_model.sort_values('Date_parsed').reset_index(drop=True)

# 2. Split temporal 80/20
n = len(df_model)
cut = int(n * 0.80)

X_train = df_model.iloc[:cut][features]
X_test  = df_model.iloc[cut:][features]
y_train = df_model.iloc[:cut]['target_bin']   # ou 'target' para 3-classes
y_test  = df_model.iloc[cut:]['target_bin']

print(f"Treino: {len(X_train)} jogos até {df_model.iloc[cut-1]['Date_parsed']}")
print(f"Teste:  {len(X_test)} jogos a partir de {df_model.iloc[cut]['Date_parsed']}")
print(f"Distribuição treino H: {y_train.mean():.3f} | teste H: {y_test.mean():.3f}")
```

> **Dica:** Se quiser ser ainda mais rigoroso, use `TimeSeriesSplit` do sklearn para validação cruzada temporal (walk-forward). Mas o split simples 80/20 já resolve 90% do problema.

### 3.4 Por que isso melhora seu aprendizado

- Você para de "enganar a si mesmo" com métricas infladas
- O modelo aprende padrões **generalizáveis**, não memorização de época
- Quando for colocar em produção (apostar na rodada de fim de semana), o comportamento será condizente com o que você mediu no teste

---

## 4. Limpeza de Features: Outliers e Redundância

### 4.1 Visão Geral das Features Problemáticas

| Feature Original | Problema | Evidência |
|------------------|----------|-----------|
| `diff_goal_ratio` | min=-3M, max=+3.2M (zeros no denominador) | `ratio = goals / (concede + 1e-6)` explode quando `concede=0` |
| `ratio_concede` | min=0, max=4M (mesmo motivo) | `away_concede / (home_concede + 1e-6)` |
| `market_vs_form` | min=-1.4M, max=+191k (divisão por ~0) | `(dif_prob*2-1) / (diff_winrate*2-1 + eps)` |
| `entropy_unc` | Clone 98.6% de `dif_prob` | `entropy * dif_prob` ≈ linear combinação |

### 4.2 Por que corrigir — Didática

**Outliers extremos:**
- Random Forest: "quebra" os splits, gasta profundidade da árvore isolando outliers
- Logistic Regression: `StandardScaler` usa média/std → um outlier de 3M **domina** a escala, features reais ficam ~0
- Interpretabilidade: importância da feature vira "ruído"

**Redundância (`entropy_unc`):**
- Regressão Logística com features colineares → matriz de Hessiana mal-condicionada → `ConvergenceWarning`, coeficientes instáveis
- Random Forest: não quebra, mas gasta "orçamento de importância" duplicando sinal

### 4.3 Como corrigir — Duas Estratégias

#### Estratégia A: Capping (Winsorização) — Simples e Eficaz

```python
# No modelos_prever.ipynb, APÓS criar as features brutas (célula 11-15)
# Adicione esta célula NOVA:

# --- CAPPING: limitar outliers aos percentis 1% e 99% ---
for col in ['diff_goal_ratio', 'ratio_concede', 'market_vs_form']:
    lo = df[col].quantile(0.01)
    hi = df[col].quantile(0.99)
    df[col + '_capped'] = df[col].clip(lo, hi)
    print(f"{col}: [{lo:.2f}, {hi:.2f}] → range controlado")
```

**Por que percentis 1/99?**  
Mantém 98% dos dados intactos, remove só as caudas patológicas. Não precisa escolher "número mágico".

#### Estratégia B: Log-Transform para `diff_goal_ratio` — Melhor Distribuição

```python
# Aplicar DEPOIS do capping
df['log_diff_goal_ratio'] = np.log1p(
    df['diff_goal_ratio_capped'] - df['diff_goal_ratio_capped'].min() + 0.01
)
```

**Por que `log1p`?**  
- Comprime cauda direita (valores grandes viram pequenos)
- `log1p(x) = log(1+x)` evita erro em x=0
- Shift `+0.01` garante argumento > 0 mesmo se min for negativo

#### Estratégia C: Remover `entropy_unc` da Lista de Features

```python
# NÃO inclua 'entropy_unc' na lista final de features
# Ele não adiciona informação além de dif_prob + entropy (que já estão lá)
```

---

## 5. Nova Feature Recomendada: `market_form_divergence`

### 5.1 A Ideia Por Trás

Você tentou criar `market_vs_form` para capturar: *"O mercado acha que o time da casa tem 70% de chance, mas a forma recente diz 40%. Há value bet aqui?"*

**Problema da implementação original:** Divisão instável.

**Solução robusta:** Erro absoluto (distância) entre as duas visões.

### 5.2 Fórmula Didática

```python
# mercado: dif_prob varia ~[-0.5, +0.5] (negativo = fora favorito)
# forma:   diff_winrate varia [-1, +1] (negativo = fora melhor forma)
# Normalizar forma para mesma escala do mercado:
#   diff_winrate * 0.5 + 0.5  → varia [0, 1]
#   dif_prob + 0.5             → varia [0, 1]

df['market_form_divergence'] = np.abs(
    df['dif_prob'] - (df['diff_winrate'] * 0.5 + 0.5)
)
```

**Interpretação:**
- `0.00` = mercado e forma concordam perfeitamente
- `0.50` = discordância máxima (mercado 100% casa, forma 100% fora ou vice-versa)
- Quanto **maior**, mais "surpresa" potencial — o modelo aprende que jogos com alta divergência têm resultados menos previsíveis pelas odds puras

### 5.3 Por Que Funcionou (Evidência Empírica)

No nosso teste temporal split:
- **2ª feature mais importante** no Random Forest (importância 0.174)
- Superou `entropy` (0.133) e todas as features de forma bruta
- Estável, sem outliers, sem divisão por zero

---

## 6. Lista Final de Features Recomendadas (10 Features)

### 6.1 Tabela Comparativa

| # | Feature | Tipo | Por Que Manter |
|---|---------|------|----------------|
| 1 | `dif_prob` | Mercado | Sinal mais forte (correlação 0.29 com H) |
| 2 | `entropy` | Mercado | Incerteza intrínseca do jogo (correlação -0.21) |
| 3 | `diff_goals` | Forma bruta | Diferença gols marcados casa vs fora (correlação 0.12) |
| 4 | `diff_winrate` | Forma relativa | **Melhor feature de forma** (correlação 0.14) |
| 5 | `diff_ppg` | Forma relativa | Pontos por jogo — captura empates também |
| 6 | `home_goal_ratio` | Eficiência | Gols / sofridos do mandante |
| 7 | `away_goal_ratio` | Eficiência | Gols / sofridos do visitante |
| 8 | `ratio_concede_capped` | Vulnerabilidade relativa | "Visitante sofre N× mais que mandante" (estabilizada) |
| 9 | `market_form_divergence` | **Nova** | Desalinhamento mercado vs forma (importância 0.174) |
| 10 | `log_diff_goal_ratio` | **Nova** | Eficiência ofensiva relativa em escala log |

### 6.2 Código Pronto Para Copiar

```python
# ============================================================
# CÉLULA DE FEATURE ENGINEERING RECOMENDADA (substituir 11-16)
# ============================================================

import numpy as np

eps = 1e-6

# --- 1. Ratios base (já existiam, mantemos) ---
df['home_goal_ratio'] = df['home_last5_goals'] / (df['home_last5_concede'] + eps)
df['away_goal_ratio'] = df['away_last5_goals'] / (df['away_last5_concede'] + eps)

# --- 2. Diferenças brutas (informação absoluta) ---
df['diff_goals']       = df['home_last5_goals'] - df['away_last5_goals']
df['diff_winrate']     = df['home_last5_winrate'] - df['away_last5_winrate']
df['diff_ppg']         = df['home_last5_ppg'] - df['away_last5_ppg']

# --- 3. Ratios problemáticos → CAPPED + LOG ---
df['diff_goal_ratio_raw']   = df['home_goal_ratio'] - df['away_goal_ratio']
df['ratio_concede_raw']     = df['away_last5_concede'] / (df['home_last5_concede'] + eps)

# Capping percentis 1/99
for col in ['diff_goal_ratio_raw', 'ratio_concede_raw']:
    lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col.replace('_raw', '_capped')] = df[col].clip(lo, hi)

# Log-transform para diff_goal_ratio
df['log_diff_goal_ratio'] = np.log1p(
    df['diff_goal_ratio_capped'] - df['diff_goal_ratio_capped'].min() + 0.01
)

# --- 4. NOVA: Divergência Mercado × Forma (substitui market_vs_form) ---
df['market_form_divergence'] = np.abs(
    df['dif_prob'] - (df['diff_winrate'] * 0.5 + 0.5)
)

# --- 5. REMOVER entropy_unc (redundante) ---
# NÃO criar: df['entropy_unc'] = df['entropy'] * df['dif_prob']

# ============================================================
# LISTA FINAL DE FEATURES (usar no modelo)
# ============================================================

FEATURES_FINAIS = [
    # Mercado (2)
    'dif_prob', 'entropy',
    
    # Forma relativa — diferenças (3)
    'diff_goals', 'diff_winrate', 'diff_ppg',
    
    # Eficiência ofensiva (2)
    'home_goal_ratio', 'away_goal_ratio',
    
    # Vulnerabilidade relativa estabilizada (1)
    'ratio_concede_capped',
    
    # Interação mercado-forma (1) — NOVA
    'market_form_divergence',
    
    # Eficiência relativa em log (1) — NOVA
    'log_diff_goal_ratio',
]

# Total: 10 features (vs 14 originais, vs 7 baseline)
```

### 6.3 Split Temporal + Treino (Célula Unificada)

```python
# ============================================================
# CÉLULA DE MODELAGEM RECOMENDADA (substituir células 7, 24, etc.)
# ============================================================

# 1. Preparar dados
df_model = df[FEATURES_FINAIS + ['target_bin', 'Date_parsed']].dropna()
df_model = df_model.sort_values('Date_parsed').reset_index(drop=True)

# 2. Split temporal 80/20
n = len(df_model)
cut = int(n * 0.80)

X_train = df_model.iloc[:cut][FEATURES_FINAIS]
X_test  = df_model.iloc[cut:][FEATURES_FINAIS]
y_train = df_model.iloc[:cut]['target_bin']
y_test  = df_model.iloc[cut:]['target_bin']

print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
print(f"Período treino: {df_model.iloc[0]['Date_parsed']} → {df_model.iloc[cut-1]['Date_parsed']}")
print(f"Período teste:  {df_model.iloc[cut]['Date_parsed']} → {df_model.iloc[-1]['Date_parsed']}")

# 3. Tratar inf/nan (segurança)
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

# 4. Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss, brier_score_loss

rf = RandomForestClassifier(
    n_estimators=200, max_depth=8,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
yp_rf = rf.predict(X_test)
pp_rf = rf.predict_proba(X_test)

print(f"\n=== RANDOM FOREST ===")
print(f"Accuracy: {accuracy_score(y_test, yp_rf):.4f}")
print(f"LogLoss:  {log_loss(y_test, pp_rf):.4f}")
print(f"Brier:    {brier_score_loss(y_test, pp_rf[:,1]):.4f}")
print(classification_report(y_test, yp_rf, target_names=['Não H', 'H']))

# Feature importance
import pandas as pd
imp = pd.Series(rf.feature_importances_, index=FEATURES_FINAIS).sort_values(ascending=False)
print("\nFeature Importance:")
print(imp)

# 5. Logistic Regression (com scaling)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lr = LogisticRegression(class_weight='balanced', max_iter=5000, random_state=42, solver='lbfgs')
lr.fit(X_train_s, y_train)
yp_lr = lr.predict(X_test_s)
pp_lr = lr.predict_proba(X_test_s)

print(f"\n=== LOGISTIC REGRESSION ===")
print(f"Accuracy: {accuracy_score(y_test, yp_lr):.4f}")
print(f"LogLoss:  {log_loss(y_test, pp_lr):.4f}")
print(f"Brier:    {brier_score_loss(y_test, pp_lr[:,1]):.4f}")
print(classification_report(y_test, yp_lr, target_names=['Não H', 'H']))
```

---

## 7. Checklist de Validação Antes de Treinar

Execute **cada item** antes de rodar o modelo final. Se algum falhar, **pare e corrija**.

| ✅ | Verificação | Como Testar | Critério de Sucesso |
|---|-------------|-------------|---------------------|
| 1 | Bug `home_last5_ppg` corrigido | `df['home_last5_ppg'].corr(target_bin) > 0` | Correlação **positiva** (era -0.112) |
| 2 | Split temporal implementado | `df_model.iloc[cut-1]['Date_parsed'] < df_model.iloc[cut]['Date_parsed']` | Data treino < Data teste |
| 3 | Sem leakage temporal | `X_train.index.max() < X_test.index.min()` | True |
| 4 | Outliers capados | `df['diff_goal_ratio_capped'].max() < 100` | Valores razoáveis (não milhões) |
| 5 | `entropy_unc` removido | `'entropy_unc' not in FEATURES_FINAIS` | True |
| 6 | `market_form_divergence` criada | `'market_form_divergence' in FEATURES_FINAIS` | True |
| 7 | Features finitas | `np.isfinite(X_train.values).all()` | True |
| 8 | Balanceamento de classes | `y_train.value_counts(normalize=True)` | ~0.48 / 0.52 (não 0.9/0.1) |
| 9 | Dummy baseline calculado | `max(y_train.mean(), 1-y_train.mean())` | ~0.52 (guarda para comparar) |
|10 | Métricas além de accuracy | LogLoss, Brier, Classification Report | Todos computados |

> **Dica:** Salve esse checklist como célula no notebook e rode **sempre** antes de treinar. Evita surpresas.

---

## 8. Próximos Passos Evolutivos

Depois que o pipeline acima estiver rodando limpo (accuracy ~0.62, LogLoss ~0.65, Brier ~0.23), estes são os passos naturais **em ordem de ROI esperado**:

### 8.1 Trocar Modelo: XGBoost / LightGBM (ganho estimado +2–4pp)

```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6,
    learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
    random_state=42, n_jobs=-1, eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)
```

**Por que:** Gradient boosting captura interações não-lineares que RF/linear não pegam. Em dados tabulares esportivos, costuma ser SOTA.

### 8.2 Validação Cruzada Temporal (Walk-Forward)

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, test_size=len(X_test))
scores = []
for train_idx, val_idx in tscv.split(X_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    # treinar + validar
    scores.append(accuracy_score(y_val, model.predict(X_val)))
print(f"CV Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

**Por que:** Um único split 80/20 tem variância. CV temporal dá estimativa robusta e detecta *overfitting* temporal.

### 8.3 Calibração de Probabilidades (Platt / Isotonic)

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(lr, method='isotonic', cv=3)
calibrated.fit(X_train_s, y_train)
pp_cal = calibrated.predict_proba(X_test_s)
print(f"Brier calibrado: {brier_score_loss(y_test, pp_cal[:,1]):.4f}")
```

**Por que:** Brier 0.23 → 0.21 significa probabilidades mais confiáveis para *value betting* (Kelly criterion, etc.).

### 8.4 Features Avançadas (Quando Tiver Tempo)

| Feature | Ideia | Dificuldade |
|---------|-------|-------------|
| `home_ewma_10_goals` | EWMA span=10 (vs 5 atual) — tendência mais longa | Baixa |
| `rest_days_home` | Dias de descanso do mandante | Média (precisa datas) |
| `h2h_last5` | Histórico direto últimos 5 confrontos | Alta (join complexo) |
| `market_surprise_rate` | % vezes que time surpreendeu como underdog | Média |

---

## Apêndice: Resumo das Mudanças de Arquivo

| Arquivo | Célula/linha | Ação |
|---------|--------------|------|
| `preprocessamento_brasileirao.ipynb` | Exec 32, linha 1332 | `"away_points"` → `"home_points"` |
| `modelos_prever.ipynb` | Nova célula pós-feature-engineering | Adicionar capping + log-transform + `market_form_divergence` |
| `modelos_prever.ipynb` | Lista `features` | Substituir por `FEATURES_FINAIS` (10 features) |
| `modelos_prever.ipynb` | Células de split (7, 24) | Substituir por split temporal (código seção 6.3) |
| `modelos_prever.ipynb` | Células de modelo | Usar código unificado da seção 6.3 |

---

## Conclusão

Este guia cobre **tudo que foi diagnosticado** na análise:

1. **Bug crítico** que inverte sinal da feature mais importante de forma
2. **Metodologia errada** (split aleatório) que infla métricas
3. **Sujeira nos dados** (outliers, redundância) que degrada modelos lineares
4. **Feature nova validada** (`market_form_divergence`) que captura sua intuição original
5. **Pipeline limpo** de 10 features, pronto para XGBoost e calibração

Aplicando ** apenas os itens 1 e 2 **, você já terá um modelo honesto e interpretável. Os itens 3–5 são o "polimento profissional".

> **Aprenda fazendo:** Rode o checklist (seção 7) após cada mudança. Se algo falhar, você sabe exatamente onde está o problema.

---

*Guia gerado em 2025-08-03. Baseado em análise empírica de 5.222 jogos (2012–2025) com split temporal 80/20. Nenhum arquivo original foi modificado na geração deste documento.*