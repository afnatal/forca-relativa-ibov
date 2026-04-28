# Força Relativa B3 x IBOV — Versão ELITE

App em Streamlit para ranking de força relativa de ativos da B3 contra o Ibovespa.

## Principais recursos

- Ranking de ativos contra o IBOV.
- Análise setorial restrita a IFNC, IMAT, ICON, UTIL e IMOB.
- Opção de carteira automática do IBOV via B3, lista manual ou upload CSV.
- Linha de força relativa `Ativo / IBOV`, normalizada em base 100.
- Indicador visual do Score com linha zero, MM20 e histograma.
- Mapa de calor das janelas relativas.
- Score Simples e Score Elite.

## Score Simples

O Score Simples é a média ponderada dos retornos relativos contra o IBOV:

`Relativo Nd % = Retorno do ativo em N pregões - Retorno do IBOV em N pregões`

Pesos padrão:

- 5 pregões: 10%
- 20 pregões: 30%
- 60 pregões: 30%
- 120 pregões: 30%
- 252 pregões: 10%

Quando uma janela não está selecionada ou não possui dados suficientes, o app recalibra usando apenas as janelas disponíveis.

## Score Elite com Sharpe + Sortino

O Score Elite pondera o Score Simples pelo fator de qualidade:

`Score Elite = Score Simples × Fator de Qualidade`

O Fator de Qualidade usa:

- 60% Sharpe 20d
- 40% Sortino 20d

Ambos são calculados com retornos logarítmicos diários, sem anualizar. O fator é limitado entre 0,25 e 2,00 para evitar distorções por outliers e para não inverter o sinal da força relativa.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```


## Atualização desta versão

A aba **Indicador Score** agora plota duas linhas:

- **Score Simples**: força relativa pura contra o IBOV.
- **Score Elite**: Score Simples ponderado pelo fator de qualidade Sharpe + Sortino.

Também foram adicionados comentários na interface explicando:

- Relativo por janela;
- Score Simples;
- Sharpe 20d;
- Sortino 20d;
- Fator de Qualidade;
- Score Elite;
- interpretação prática do gráfico.
