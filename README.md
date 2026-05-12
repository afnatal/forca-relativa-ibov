# Força Relativa B3 x Benchmarks — Versão ELITE

App em Streamlit para ranking de força relativa de ativos da B3 contra benchmarks configuráveis,
com Score Elite ajustado por Sharpe e Sortino descontados pelo CDI.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Novidades desta versão

| Melhoria | Descrição |
|---|---|
| **CDI/Selic configurável** | Sharpe e Sortino agora descontam o custo de oportunidade real do mercado brasileiro. Ajuste na barra lateral. |
| **Download em lote** | Tenta baixar todos os tickers em uma única requisição (`threads=True`) antes de recorrer a aliases individuais. Muito mais rápido para o universo completo do IBOV (~70 tickers). |
| **Pesos corrigidos** | `SCORE_WEIGHTS` agora somam exatamente **1,00** (era 1,10 na versão anterior). |
| **`ffill(limit=3)`** | Preenchimento máximo de 3 dias consecutivos de dados faltantes. Evita propagar preços obsoletos em suspensões ou circuit breakers. |
| **Avisos de gaps** | A interface exibe alerta para ativos com buracos de preço superiores a 3 pregões antes do ffill. |
| **`min_periods` consistente** | `rolling_quality_factor` e `rolling_sortino_factor` agora usam `min_periods` igual em todas as janelas rolling, eliminando assimetria entre Sharpe e Sortino nas primeiras observações do histórico. |

---

## Principais recursos

- Ranking de ativos contra benchmark configurável (IBOV, SPX, NASDAQ, DXY ou ticker personalizado).
- Análise setorial restrita a IFNC, IMAT, ICON, UTIL e IMOB.
- Fonte de ativos: carteira automática do IBOV via B3, lista manual ou upload CSV.
- Linha de força relativa `Ativo / Benchmark` normalizada em base 100.
- Indicador visual do Score (histograma, MM20, Score Curto Elite, Score Sortino).
- Mapa de calor das janelas relativas.
- Análise multi-benchmark simultânea (IBOV, SPX, NASDAQ e DXY).

---

## Score Simples

Média ponderada dos retornos relativos contra o benchmark:

`Relativo Nd % = Retorno do ativo em N pregões − Retorno do benchmark em N pregões`

### Pesos por janela (somam 1,00)

| Janela     | Peso |
|-----------:|-----:|
| 5 pregões  |  10% |
| 20 pregões |  25% |
| 60 pregões |  30% |
| 120 pregões | 25% |
| 252 pregões | 10% |

Quando uma janela não está selecionada ou sem dados, o Score é renormalizado automaticamente
pela divisão por `used_weight` (soma dos pesos das janelas efetivamente calculadas).

---

## Score Elite

`Score Elite = Score Simples × Fator de Qualidade`

O **Fator de Qualidade** combina:
- 60% Sharpe 20d
- 40% Sortino 20d

Ambos calculados com retornos logarítmicos diários **descontados pelo CDI** configurado na sidebar.
O fator é limitado entre **0,25×** e **2,00×** para evitar inversão de sinal por outliers.

### Por que descontar o CDI?

Em um ambiente de juros altos (Selic ≥ 10% a.a.), ativos que apenas acompanham a taxa básica
não deveriam ser premiados no Fator de Qualidade. O excesso de retorno sobre o CDI é o
componente informativo para o Sharpe e o Sortino.

`rf_daily = (1 + CDI_anual) ^ (1/252) − 1`

---

## Score Sortino puro

`Score Sortino = Score Simples × Fator Sortino`

`Fator Sortino = clip(1 + Sortino 20d / 2, 0,25, 2,00)`

Leitura defensiva que penaliza apenas volatilidade negativa.
Útil para swing trade e carrego com opções.

---

## Score Curto 5/20

`Score Curto = 0,30 × Relativo 5d + 0,70 × Relativo 20d`

Versão tática do Score para timing de entrada e saída. Vira antes do Score estrutural.

`Score Curto Elite = Score Curto × Fator de Qualidade`

---

## Download de preços

### Estratégia em duas etapas

1. **Lote (`threads=True`)** — baixa todos os tickers em uma única chamada ao Yahoo Finance.
   Reduz o tempo de carga para carteiras grandes de minutos para segundos.
2. **Aliases individuais** — para tickers que falharam no lote, tenta cada alias listado
   em `YAHOO_TICKER_ALIASES` (ex.: `IFNC.SA` → `^IFNC`).

### Qualidade dos dados

- Gaps (NaN consecutivos) são detectados **antes** do preenchimento.
- `ffill(limit=3)` propaga no máximo 3 pregões.
- Tickers com gap > 3 dias geram aviso visível na interface.

---

## Benchmarks disponíveis

| Nome               | Ticker Yahoo |
|--------------------|-------------|
| IBOV               | `^BVSP`     |
| SPX                | `^SPX`      |
| NASDAQ             | `^IXIC`     |
| DXY (Dollar Index) | `DX-Y.NYB`  |

Qualquer ticker válido do Yahoo Finance pode ser usado como benchmark personalizado.

---

## Regime de força relativa

| Regime              | Critério                                          |
|---------------------|--------------------------------------------------|
| Liderança relativa  | Rel. 20d > 0, Rel. 60d > 0 e RS > MM20          |
| Virando para cima   | Rel. 20d > 0, Rel. 60d < 0 e RS > MM20          |
| Perdendo força      | Rel. 20d < 0 e Rel. 60d > 0                     |
| Underperform        | Rel. 20d < 0, Rel. 60d < 0 e RS < MM20          |
| Neutro              | Critérios mistos ou dados insuficientes          |

---

## Requisitos

```
streamlit>=1.36
pandas>=2.0
numpy>=1.24
yfinance>=0.2.40
plotly>=5.20
requests>=2.31
```
