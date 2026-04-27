# Força Relativa B3 x IBOV

App Streamlit para ranking de força relativa de ativos da B3 contra o IBOV e análise setorial restrita aos índices:

- IFNC
- IMAT
- ICON
- UTIL
- IMOB

## Correções desta versão

- A análise setorial permanece restrita aos cinco índices definidos, com remoção definitiva do IEEX.
- O app passou a tentar tickers alternativos automaticamente quando o Yahoo Finance não retorna dados para algum índice, quando o Yahoo Finance não retorna dados para algum índice.
- Um ticker setorial indisponível não interrompe mais a execução: ele é ignorado e exibido em aviso.
- Substituição de `use_container_width=True` por `width="stretch"` para compatibilidade com versões futuras do Streamlit.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Observação

Os dados de preços vêm do Yahoo Finance via `yfinance`. Alguns índices setoriais da B3 podem ficar indisponíveis ou mudar de ticker na base do Yahoo. Por isso o app possui fallback automático e opção de edição manual dos tickers setoriais.

## Atualização: Indicador visual de Score

A aba **Indicador Score** apresenta um painel visual no estilo de indicador técnico:

- Linha do **Score Relativo** contra o IBOV;
- Linha zero;
- Média móvel de 20 períodos do Score;
- Histograma colorido conforme a direção e o sinal do Score:
  - Score > 0 e subindo: liderança relativa forte;
  - Score > 0 e caindo: liderança perdendo tração;
  - Score < 0 e caindo: underperformance aumentando;
  - Score < 0 e subindo: possível recuperação relativa.

O cálculo é feito pela média ponderada dos retornos relativos selecionados nas janelas do app.
