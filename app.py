import base64
import contextlib
import io
import json
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Força Relativa B3 x Benchmarks", layout="wide")

B3_PORTFOLIO_URL = "https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/GetPortfolioDay/{payload}"

SECTOR_INDICES = {
    "IFNC - Financeiro": "IFNC.SA",
    "IMAT - Materiais Básicos": "IMAT.SA",
    "ICON - Consumo": "ICON.SA",
    "UTIL - Utilidade Pública": "UTIL.SA",
    "IMOB - Imobiliário": "IMOB.SA",
}

YAHOO_TICKER_ALIASES = {
    "IFNC.SA": ["IFNC.SA", "^IFNC", "IFNC"],
    "IMAT.SA": ["IMAT.SA", "^IMAT", "IMAT"],
    "ICON.SA": ["ICON.SA", "^ICON", "ICON"],
    "UTIL.SA": ["UTIL.SA", "^UTIL", "UTIL"],
    "IMOB.SA": ["IMOB.SA", "^IMOB", "IMOB"],
    "^BVSP": ["^BVSP", "IBOV.SA"],
    "^SPX": ["^SPX", "^GSPC"],
    "^IXIC": ["^IXIC"],
    "DX-Y.NYB": ["DX-Y.NYB", "^DXY"],
}

BENCHMARK_OPTIONS = {
    "IBOV": "^BVSP",
    "SPX": "^SPX",
    "NASDAQ": "^IXIC",
    "DXY (Dollar Index)": "DX-Y.NYB",
}

MULTI_BENCHMARK_OPTIONS = BENCHMARK_OPTIONS.copy()

SHORT_SCORE_WEIGHTS = {5: 0.30, 20: 0.70}


def classify_short_signal(score_elite, score_short):
    """Classifica a leitura tática do Score Curto contra o Score Elite estrutural."""
    if pd.isna(score_elite) or pd.isna(score_short):
        return "Sem dados"
    if score_elite > 0 and score_short > 0:
        return "Timing favorável"
    if score_elite > 0 and score_short <= 0:
        return "Pullback / perda tática"
    if score_elite <= 0 and score_short > 0:
        return "Reversão inicial"
    return "Fraco no curto"


DEFAULT_MANUAL_ASSETS = "PETR4, VALE3, ITUB4, BBAS3, BBDC4, BPAC11, AXIA3, PRIO3, WEGE3, SBSP3"

FALLBACK_IBOV = [
    "ABEV3", "ASAI3", "AURE3", "AZUL4", "B3SA3", "BBAS3", "BBDC3", "BBDC4",
    "BBSE3", "BEEF3", "BPAC11", "BRAP4", "BRFS3", "BRKM5", "CCRO3", "CMIG4",
    "COGN3", "CPFE3", "CPLE6", "CRFB3", "CSAN3", "CSNA3", "CYRE3", "ELET3",
    "ELET6", "EMBR3", "ENEV3", "ENGI11", "EQTL3", "EZTC3", "FLRY3", "GGBR4",
    "GOAU4", "HAPV3", "HYPE3", "IGTI11", "IRBR3", "ITSA4", "ITUB4", "JBSS3",
    "KLBN11", "LREN3", "MGLU3", "MRFG3", "MRVE3", "MULT3", "NTCO3", "PETR3",
    "PETR4", "PRIO3", "RADL3", "RAIL3", "RAIZ4", "RENT3", "RRRP3", "SANB11",
    "SBSP3", "SLCE3", "SMTO3", "SUZB3", "TAEE11", "TIMS3", "TOTS3", "UGPA3",
    "USIM5", "VALE3", "VAMO3", "VBBR3", "VIVA3", "WEGE3", "YDUQ3",
]


def to_b3_payload(index: str, page: int = 1, page_size: int = 200) -> str:
    data = {"language": "pt-br", "pageNumber": page, "pageSize": page_size, "index": index.upper(), "segment": "1"}
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(raw).decode("utf-8")


@st.cache_data(ttl=60 * 60)
def fetch_b3_index_portfolio(index: str) -> pd.DataFrame:
    payload = to_b3_payload(index)
    url = B3_PORTFOLIO_URL.format(payload=payload)
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()
    rows = data.get("results", []) or data.get("Results", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.columns = [str(c).lower() for c in df.columns]
    if "cod" in df.columns:
        df["ticker"] = df["cod"].astype(str).str.strip()
    elif "code" in df.columns:
        df["ticker"] = df["code"].astype(str).str.strip()
    else:
        df["ticker"] = df[df.columns[0]].astype(str).str.strip()
    return df[df["ticker"].str.len() > 0].drop_duplicates("ticker")


def br_to_yahoo(tickers: List[str]) -> List[str]:
    out = []
    for t in tickers:
        t = str(t).strip().upper()
        if not t:
            continue
        if t.startswith("^") or "." in t:
            out.append(t)
        else:
            out.append(f"{t}.SA")
    return sorted(set(out))


def yahoo_to_br(ticker: str) -> str:
    return ticker.replace(".SA", "").replace("^", "")


@st.cache_data(ttl=60 * 30)
def download_prices(tickers: Tuple[str, ...], start: date, end: date) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    if not tickers:
        return pd.DataFrame(), {}, []
    frames = []
    used_map: Dict[str, str] = {}
    failed: List[str] = []
    for original in tickers:
        original = str(original).strip()
        if not original:
            continue
        candidates = YAHOO_TICKER_ALIASES.get(original, [original])
        series = None
        used = None
        for candidate in candidates:
            try:
                buf_out, buf_err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                    data = yf.download(
                        candidate,
                        start=start.isoformat(),
                        end=(end + timedelta(days=1)).isoformat(),
                        auto_adjust=True,
                        progress=False,
                        threads=False,
                    )
                if data is None or data.empty:
                    continue
                if isinstance(data.columns, pd.MultiIndex):
                    close = data["Close"].iloc[:, 0] if "Close" in data.columns.get_level_values(0) else None
                else:
                    close = data.get("Close")
                if close is None or close.dropna().empty:
                    continue
                series = close.rename(original)
                used = candidate
                break
            except Exception:
                continue
        if series is not None:
            frames.append(series)
            used_map[original] = used or original
        else:
            failed.append(original)
    if not frames:
        return pd.DataFrame(), used_map, failed
    close = pd.concat(frames, axis=1).dropna(how="all").ffill()
    return close, used_map, failed



def calc_quality_metrics(asset: pd.Series, rf_daily: float = 0.0, window: int = 20) -> Dict[str, float]:
    """Calcula Sharpe, Sortino e fator de qualidade para ponderar o Score.

    Usa retornos logarítmicos diários na janela de 20 pregões. Para ranking,
    não anualiza, pois todos os ativos são comparados no mesmo horizonte.
    """
    log_ret = np.log(asset / asset.shift(1)).dropna()
    recent = log_ret.tail(window)
    if len(recent) < max(5, window // 2):
        return {"Sharpe 20d": np.nan, "Sortino 20d": np.nan, "Qualidade S/S": np.nan}

    excess = recent - rf_daily
    vol = excess.std(ddof=0)
    downside = excess[excess < 0]
    down_vol = downside.std(ddof=0) if len(downside) >= 2 else np.nan

    sharpe = excess.mean() / vol if pd.notna(vol) and vol > 0 else np.nan
    sortino = excess.mean() / down_vol if pd.notna(down_vol) and down_vol > 0 else np.nan
    if pd.isna(sortino) and pd.notna(sharpe):
        sortino = sharpe

    quality_raw = 0.6 * sharpe + 0.4 * sortino if pd.notna(sharpe) and pd.notna(sortino) else np.nan
    quality_factor = np.clip(1 + (quality_raw / 2), 0.25, 2.00) if pd.notna(quality_raw) else np.nan
    return {"Sharpe 20d": sharpe, "Sortino 20d": sortino, "Qualidade S/S": quality_factor}


def rolling_quality_factor(asset: pd.Series, rf_daily: float = 0.0, window: int = 20) -> pd.Series:
    """Série histórica do fator de qualidade baseado em Sharpe + Sortino."""
    log_ret = np.log(asset / asset.shift(1))
    excess = log_ret - rf_daily
    mean = excess.rolling(window).mean()
    vol = excess.rolling(window).std(ddof=0)
    sharpe = mean / vol.replace(0, np.nan)

    downside = excess.where(excess < 0)
    down_vol = downside.rolling(window, min_periods=max(5, window // 2)).std(ddof=0)
    sortino = mean / down_vol.replace(0, np.nan)
    sortino = sortino.fillna(sharpe)

    quality_raw = 0.6 * sharpe + 0.4 * sortino
    quality_factor = (1 + quality_raw / 2).clip(lower=0.25, upper=2.00)
    return quality_factor
def calc_metrics(prices: pd.DataFrame, benchmark: str, windows: List[int], ma_window: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prices = prices.dropna(how="all").ffill()
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark {benchmark} não encontrado nos preços baixados.")
    bench = prices[benchmark]
    tickers = [c for c in prices.columns if c != benchmark]
    rows = []
    rs_curves = pd.DataFrame(index=prices.index)
    for tk in tickers:
        s = prices[tk].dropna()
        aligned = pd.concat([s, bench], axis=1, join="inner").dropna()
        if len(aligned) < max(windows) + 2:
            continue
        asset = aligned.iloc[:, 0]
        bm = aligned.iloc[:, 1]
        rs = asset / bm
        rs_norm = rs / rs.iloc[0] * 100
        rs_curves[tk] = rs_norm.reindex(prices.index)
        row = {"Ativo": yahoo_to_br(tk)}
        score = 0.0
        short_score = 0.0
        short_used_weight = 0.0
        rel_by_window = {}
        weights = {5: 0.10, 20: 0.30, 60: 0.30, 120: 0.30, 252: 0.10}
        used_weight = 0.0
        for w in windows:
            if len(asset) > w:
                ret_asset = asset.iloc[-1] / asset.iloc[-w - 1] - 1
                ret_bench = bm.iloc[-1] / bm.iloc[-w - 1] - 1
                rel = ret_asset - ret_bench
                rel_by_window[w] = rel
                row[f"Ativo {w}d %"] = ret_asset * 100
                row[f"Benchmark {w}d %"] = ret_bench * 100
                row[f"Relativo {w}d %"] = rel * 100
                wt = weights.get(w, 1 / len(windows))
                score += rel * wt
                used_weight += wt
                if w in SHORT_SCORE_WEIGHTS:
                    short_score += rel * SHORT_SCORE_WEIGHTS[w]
                    short_used_weight += SHORT_SCORE_WEIGHTS[w]
        rs_ma = rs.rolling(ma_window).mean()
        row["RS Atual"] = rs_norm.iloc[-1]
        row["RS x MM20 %"] = ((rs.iloc[-1] / rs_ma.iloc[-1] - 1) * 100) if len(rs_ma.dropna()) else np.nan
        row["Score Simples"] = (score / used_weight * 100) if used_weight else np.nan
        row["Score Curto 5/20"] = (short_score / short_used_weight * 100) if short_used_weight else np.nan
        q = calc_quality_metrics(asset, rf_daily=0.0, window=20)
        row.update(q)
        row["Score Elite"] = row["Score Simples"] * row["Qualidade S/S"] if pd.notna(row["Score Simples"]) and pd.notna(row["Qualidade S/S"]) else np.nan
        row["Score Curto Elite"] = row["Score Curto 5/20"] * row["Qualidade S/S"] if pd.notna(row["Score Curto 5/20"]) and pd.notna(row["Qualidade S/S"]) else np.nan
        row["Sinal Curto"] = classify_short_signal(row["Score Elite"], row["Score Curto Elite"])
        row["Score"] = row["Score Elite"]
        row["Regime"] = classify_regime(row)
        pfr = classify_loss_of_strength(row)
        row["Alerta PFR"] = pfr["Alerta PFR"]
        row["PFR Pontos"] = pfr["PFR Pontos"]
        rows.append(row)
    ranking = pd.DataFrame(rows).sort_values("Score Elite", ascending=False) if rows else pd.DataFrame()
    return ranking, rs_curves


def calc_score_series(prices: pd.DataFrame, asset_col: str, benchmark_col: str, windows: List[int], ma_window: int = 20) -> pd.DataFrame:
    """Calcula a série histórica do Score Relativo para um ativo/índice contra o benchmark."""
    if asset_col not in prices.columns or benchmark_col not in prices.columns:
        return pd.DataFrame()
    aligned = prices[[asset_col, benchmark_col]].dropna().ffill()
    if aligned.empty:
        return pd.DataFrame()

    asset = aligned[asset_col]
    bench = aligned[benchmark_col]
    weights = {5: 0.10, 20: 0.30, 60: 0.30, 120: 0.30, 252: 0.10}
    score = pd.Series(0.0, index=aligned.index)
    total_weight = pd.Series(0.0, index=aligned.index)

    for w in windows:
        if w <= 0:
            continue
        rel = asset.pct_change(w) - bench.pct_change(w)
        wt = weights.get(w, 1 / max(len(windows), 1))
        score = score.add(rel.fillna(0) * wt, fill_value=0)
        total_weight = total_weight.add(rel.notna().astype(float) * wt, fill_value=0)

    score_pct = (score / total_weight.replace(0, np.nan)) * 100

    # Score Curto 5/20: versão tática, voltada para timing de entrada/saída.
    # Usa somente a força relativa de 5 e 20 pregões, com maior peso para 20d.
    short_score = pd.Series(0.0, index=aligned.index)
    short_total_weight = pd.Series(0.0, index=aligned.index)
    for w, wt in SHORT_SCORE_WEIGHTS.items():
        rel = asset.pct_change(w) - bench.pct_change(w)
        short_score = short_score.add(rel.fillna(0) * wt, fill_value=0)
        short_total_weight = short_total_weight.add(rel.notna().astype(float) * wt, fill_value=0)
    short_score_pct = (short_score / short_total_weight.replace(0, np.nan)) * 100

    quality_factor = rolling_quality_factor(asset, rf_daily=0.0, window=20)
    df = pd.DataFrame(index=aligned.index)
    df["Score Simples"] = score_pct
    df["Score Curto 5/20"] = short_score_pct
    df["Qualidade S/S"] = quality_factor
    df["Score Elite"] = df["Score Simples"] * df["Qualidade S/S"]
    df["Score Curto Elite"] = df["Score Curto 5/20"] * df["Qualidade S/S"]
    df["Score"] = df["Score Elite"]
    df["MM20 Score"] = df["Score"].rolling(ma_window).mean()
    df["Histograma"] = df["Score"] - df["MM20 Score"]
    df["Direção"] = df["Score"].diff()

    # Indicador de Perda de Força Relativa (PFR).
    # Ele busca situações em que o ativo ainda tem uma leitura de força no horizonte maior,
    # mas começa a perder tração nas janelas mais recentes e/ou perde a média da própria RS.
    df["Relativo 5d %"] = (asset.pct_change(5) - bench.pct_change(5)) * 100
    df["Relativo 20d %"] = (asset.pct_change(20) - bench.pct_change(20)) * 100
    df["Relativo 60d %"] = (asset.pct_change(60) - bench.pct_change(60)) * 100
    rs = asset / bench
    rs_ma = rs.rolling(ma_window).mean()
    df["RS x MM20 %"] = (rs / rs_ma - 1) * 100
    df["PFR Pontos"] = calc_loss_of_strength_points_series(df)
    df["Alerta PFR"] = df["PFR Pontos"].apply(label_loss_of_strength)

    def classify_bar(row):
        if pd.isna(row["Score"]) or pd.isna(row["Direção"]):
            return "Neutro"
        if row["Score"] >= 0 and row["Direção"] >= 0:
            return "Score > 0 e subindo"
        if row["Score"] >= 0 and row["Direção"] < 0:
            return "Score > 0 e caindo"
        if row["Score"] < 0 and row["Direção"] < 0:
            return "Score < 0 e caindo"
        return "Score < 0 e subindo"

    df["Condição"] = df.apply(classify_bar, axis=1)
    return df.dropna(subset=["Score"])


def enable_horizontal_zoom(fig: go.Figure, range_slider: bool = True) -> go.Figure:
    """Ativa controles úteis para zoom/pan no eixo horizontal dos gráficos temporais."""
    fig.update_xaxes(
        rangeslider=dict(visible=range_slider),
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1a", step="year", stepmode="backward"),
                dict(step="all", label="Tudo"),
            ])
        ),
        type="date",
    )
    fig.update_layout(dragmode="pan")
    return fig


def plotly_time_chart(fig: go.Figure, key: str):
    """Renderiza gráfico Plotly com barra de ferramentas ativa para zoom horizontal."""
    st.plotly_chart(
        fig,
        width="stretch",
        key=key,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
        },
    )


def render_score_indicator(prices: pd.DataFrame, asset_col: str, benchmark_col: str, windows: List[int], title: str):
    score_df = calc_score_series(prices, asset_col, benchmark_col, windows)
    if score_df.empty:
        st.warning("Sem dados suficientes para calcular o indicador visual de Score.")
        return

    color_map = {
        "Score > 0 e subindo": "#00A650",
        "Score > 0 e caindo": "#8FD19E",
        "Score < 0 e caindo": "#D62728",
        "Score < 0 e subindo": "#F4A6A6",
        "Neutro": "#B0B0B0",
    }
    bar_colors = [color_map.get(c, "#B0B0B0") for c in score_df["Condição"]]

    # O histograma usa o Score Elite porque ele é o score final do modelo.
    # O gráfico foi mantido mais limpo: não exibe mais o Score Simples
    # nem o Score Curto 5/20 puro, preservando o foco no Score Elite,
    # no Score Curto Elite, na MM20 do Score Elite e no preço em eixo secundário.
    # A linha de fechamento do ativo é plotada no eixo Y secundário para comparar
    # a evolução do preço com a evolução da força relativa em cada dia.
    close_series = prices[asset_col].reindex(score_df.index).dropna() if asset_col in prices.columns else pd.Series(dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=score_df.index,
        y=score_df["Score Elite"],
        name="Histograma Score Elite",
        marker_color=bar_colors,
        opacity=0.45,
        hovertemplate="Data=%{x}<br>Score Elite=%{y:.2f}%<extra></extra>",
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=score_df.index,
        y=score_df["Score Curto Elite"],
        mode="lines",
        name="Score Curto Elite",
        line=dict(width=2),
        opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=score_df.index,
        y=score_df["Score Elite"],
        mode="lines",
        name="Score Elite (Sharpe + Sortino)",
        line=dict(width=3),
    ))
    fig.add_trace(go.Scatter(
        x=score_df.index,
        y=score_df["MM20 Score"],
        mode="lines",
        name="MM20 do Score Elite",
        line=dict(width=2, dash="dot"),
        yaxis="y",
    ))
    if not close_series.empty:
        fig.add_trace(go.Scatter(
            x=close_series.index,
            y=close_series,
            mode="lines",
            name=f"Fechamento diário — {yahoo_to_br(asset_col)}",
            line=dict(width=2),
            opacity=0.85,
            yaxis="y2",
            hovertemplate="Data=%{x}<br>Fechamento=%{y:.2f}<extra></extra>",
        ))

    # Marcadores do Indicador de Perda de Força Relativa (PFR).
    pfr_mask = score_df["Alerta PFR"].isin(["Atenção: perdendo força", "Perda confirmada"])
    if pfr_mask.any():
        pfr_colors = score_df.loc[pfr_mask, "Alerta PFR"].map({
            "Atenção: perdendo força": "#F28E2B",
            "Perda confirmada": "#D62728",
        }).fillna("#F28E2B")
        fig.add_trace(go.Scatter(
            x=score_df.index[pfr_mask],
            y=score_df.loc[pfr_mask, "Score Elite"],
            mode="markers",
            name="Alerta PFR",
            marker=dict(size=11, symbol="triangle-down", color=pfr_colors, line=dict(width=1, color="#333333")),
            hovertemplate=(
                "Data=%{x}<br>Score Elite=%{y:.2f}%<br>"
                "PFR=%{customdata[0]}<br>Pontos=%{customdata[1]:.0f}<extra></extra>"
            ),
            customdata=np.column_stack([
                score_df.loc[pfr_mask, "Alerta PFR"],
                score_df.loc[pfr_mask, "PFR Pontos"],
            ]),
        ))
    fig.add_hline(y=0, line_width=1, line_dash="dash", annotation_text="Zero", annotation_position="bottom right")
    fig.update_layout(
        title=title,
        yaxis=dict(title="Score relativo (%)", side="left"),
        yaxis2=dict(
            title=f"Preço de fechamento — {yahoo_to_br(asset_col)}",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        xaxis_title="Data",
        legend_title="Indicador",
        hovermode="x unified",
        height=650,
    )
    enable_horizontal_zoom(fig, range_slider=True)
    plotly_time_chart(fig, key=f"score_indicator_{asset_col}_{benchmark_col}")

    last = score_df.iloc[-1]
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Score Simples", f"{last['Score Simples']:.2f}%" if pd.notna(last["Score Simples"]) else "n/d")
    c2.metric("Score Curto", f"{last['Score Curto 5/20']:.2f}%" if pd.notna(last["Score Curto 5/20"]) else "n/d")
    c3.metric("Fator Qualidade", f"{last['Qualidade S/S']:.2f}x" if pd.notna(last["Qualidade S/S"]) else "n/d")
    c4.metric("Score Elite", f"{last['Score Elite']:.2f}%" if pd.notna(last["Score Elite"]) else "n/d")
    c5.metric("Score Curto Elite", f"{last['Score Curto Elite']:.2f}%" if pd.notna(last["Score Curto Elite"]) else "n/d")
    c6.metric("MM20 Elite", f"{last['MM20 Score']:.2f}%" if pd.notna(last["MM20 Score"]) else "n/d")
    c7.metric("Condição", str(last["Condição"]))
    c8.metric("Alerta PFR", str(last["Alerta PFR"]))

    st.markdown(
        """
**Como ler este gráfico:**

- **Score Simples**: força relativa pura do ativo contra o IBOV nas janelas selecionadas.
- **Score Elite**: Score Simples ponderado pelo **Fator de Qualidade**, calculado com Sharpe 20d e Sortino 20d.
- **Score Curto 5/20**: versão tática do Score, usando somente 5 e 20 pregões; serve para timing e vira antes do Score completo.
- **Score Curto Elite**: Score Curto 5/20 ponderado pelo mesmo Fator de Qualidade.
- **MM20 do Score Elite**: média móvel de 20 pregões do Score Elite; ajuda a identificar consistência ou perda de força.
- **Histograma**: barras do Score Elite coloridas conforme o sinal e a direção do score.
- **Fechamento diário**: preço de fechamento do ativo no eixo secundário à direita; permite comparar se o preço está confirmando, antecipando ou divergindo da força relativa.
- **Alerta PFR**: triângulos no gráfico indicam perda de força relativa quando o ativo ainda tem relativo 60d positivo, mas começa a perder tração em 20d/5d, perde a MM20 da RS ou tem Score Elite negativo.

Quando o **Score Elite** está acima de zero e acima da MM20, o ativo está em liderança relativa com melhor qualidade de retorno.
Quando o **Score Simples** sobe, mas o **Score Elite** não acompanha, o ativo pode estar subindo com pior relação retorno/risco.
Quando o **Score Curto** melhora antes do **Score Elite**, pode ser sinal inicial de rotação positiva; quando piora com Score Elite ainda positivo, pode indicar pullback ou perda tática de força.
        """
    )


def label_loss_of_strength(points: float) -> str:
    """Classifica o Indicador de Perda de Força Relativa (PFR)."""
    if pd.isna(points):
        return "Sem dados"
    if points >= 4:
        return "Perda confirmada"
    if points >= 2:
        return "Atenção: perdendo força"
    if points >= 1:
        return "Monitorar"
    return "Sem alerta"


def classify_loss_of_strength(row: Dict) -> Dict[str, object]:
    """Calcula o alerta atual de Perda de Força Relativa para a tabela de ranking.

    A lógica procura uma situação típica de liderança passada com perda de tração:
    o relativo de 60 dias ainda é positivo, mas os períodos mais curtos começam a
    enfraquecer, a linha RS perde a MM20 ou o Score Elite fica negativo.
    """
    r5 = row.get("Relativo 5d %", np.nan)
    r20 = row.get("Relativo 20d %", np.nan)
    r60 = row.get("Relativo 60d %", np.nan)
    rs_mm = row.get("RS x MM20 %", np.nan)
    score = row.get("Score Elite", row.get("Score", np.nan))

    if pd.isna(r60) or r60 <= 0:
        points = 0
    else:
        points = 0
        if pd.notna(r20) and r20 < 0:
            points += 2
        if pd.notna(r5) and r5 < 0:
            points += 1
        if pd.notna(rs_mm) and rs_mm < 0:
            points += 1
        if pd.notna(score) and score < 0:
            points += 1

    return {"PFR Pontos": points, "Alerta PFR": label_loss_of_strength(points)}


def calc_loss_of_strength_points_series(df: pd.DataFrame) -> pd.Series:
    """Calcula pontuação histórica do PFR para o gráfico do Indicador Score."""
    points = pd.Series(0, index=df.index, dtype=float)
    base = df["Relativo 60d %"] > 0
    points += (base & (df["Relativo 20d %"] < 0)).astype(float) * 2
    points += (base & (df["Relativo 5d %"] < 0)).astype(float) * 1
    points += (base & (df["RS x MM20 %"] < 0)).astype(float) * 1
    points += (base & (df["Score Elite"] < 0)).astype(float) * 1
    return points


def render_loss_of_strength_explanation():
    """Explica o Indicador de Perda de Força Relativa."""
    with st.expander("Indicador de Perda de Força Relativa (PFR)", expanded=False):
        st.markdown(
            """
O **PFR** foi criado para identificar a situação que você observou: o ativo teve liderança no período maior, mas começa a lateralizar ou perder tração enquanto o benchmark continua andando.

### Objetivo

Detectar **liderança passada com enfraquecimento recente**. Não é o mesmo que underperformance estrutural. O foco é encontrar ativos que ainda parecem fortes olhando o gráfico acumulado, mas já estão perdendo força relativa nas janelas mais curtas.

### Critérios usados

O PFR só é ativado quando o ativo ainda tem **Relativo 60d positivo**. A partir disso, soma pontos de alerta:

| Critério | Pontos | Interpretação |
|---|---:|---|
| Relativo 20d < 0 | +2 | O ativo começou a perder para o benchmark na janela principal de swing. |
| Relativo 5d < 0 | +1 | A perda de força apareceu também no curtíssimo prazo. |
| RS abaixo da MM20 | +1 | A linha de força relativa perdeu sua média curta. |
| Score Elite < 0 | +1 | A força relativa ajustada por Sharpe/Sortino ficou negativa. |

### Classificação

| Alerta PFR | Pontuação | Leitura |
|---|---:|---|
| **Sem alerta** | 0 | Sem perda relevante de força. |
| **Monitorar** | 1 | Primeiro sinal de enfraquecimento. |
| **Atenção: perdendo força** | 2 a 3 | Perda de tração relevante. |
| **Perda confirmada** | 4 ou mais | Divergência forte entre liderança anterior e força atual. |

### Uso prático

- **RS alto + PFR alto**: ativo ainda pode parecer bom no gráfico, mas o fluxo relativo está enfraquecendo.
- **PFR subindo**: cuidado com compras novas e com carrego por opções.
- **PFR caindo após consolidação**: pode indicar retomada de força.
            """
        )


def classify_regime(row: Dict) -> str:
    r20 = row.get("Relativo 20d %", np.nan)
    r60 = row.get("Relativo 60d %", np.nan)
    rs_mm = row.get("RS x MM20 %", np.nan)
    if pd.notna(r20) and pd.notna(r60) and pd.notna(rs_mm):
        if r20 > 0 and r60 > 0 and rs_mm > 0:
            return "Liderança relativa"
        if r20 > 0 and r60 < 0 and rs_mm > 0:
            return "Virando para cima"
        if r20 < 0 and r60 > 0:
            return "Perdendo força"
        if r20 < 0 and r60 < 0 and rs_mm < 0:
            return "Underperform"
    return "Neutro"



REGIME_COLOR_MAP = {
    "Perdendo força": "#F28E2B",      # laranja
    "Virando para cima": "#8CD17D",   # verde claro
    "Liderança relativa": "#006400",  # verde escuro
    "Underperform": "#D62728",        # vermelho
    "Neutro": "#F1C40F",              # amarelo
}


def reorder_ranking_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coloca Regime como segunda coluna, logo após Ativo/Ticker/Índice, mantendo as demais colunas."""
    if df.empty or "Regime" not in df.columns:
        return df
    first_col = None
    for candidate in ["Ativo", "Ticker", "Índice"]:
        if candidate in df.columns:
            first_col = candidate
            break
    if first_col is None:
        return df
    preferred = [first_col, "Regime"]
    if "Alerta PFR" in df.columns:
        preferred.append("Alerta PFR")
    ordered = preferred + [c for c in df.columns if c not in preferred]
    return df[ordered]

def parse_manual_list(text: str) -> List[str]:
    tokens = text.replace(";", ",").replace("\n", ",").split(",")
    return [t.strip().upper() for t in tokens if t.strip()]


def render_score_explanation(context: str = "ranking"):
    """Mostra uma explicação curta e consistente sobre o cálculo do Score."""
    with st.expander("Como o Score é calculado", expanded=False):
        st.markdown(
            """
O **Score** é uma média ponderada da performance relativa do ativo contra o benchmark selecionado.

**Performance relativa de cada janela:**

`Relativo Nd % = Retorno do ativo em N pregões - Retorno do benchmark em N pregões`

### Tipos de Score usados no app

| Indicador | Cálculo | Para que serve |
|---|---|---|
| **Relativo Nd %** | `Retorno do ativo em N pregões - Retorno do benchmark em N pregões` | Mede se o ativo ganhou ou perdeu do IBOV em cada janela. |
| **Score Simples** | Média ponderada dos retornos relativos disponíveis | Mede força relativa pura, sem ajuste de risco. |
| **Score Curto 5/20** | `0,3 × RS 5d + 0,7 × RS 20d` | Mede momentum relativo tático para timing de entrada/saída. |
| **Sharpe 20d** | Retorno médio excedente / volatilidade total dos retornos | Mede consistência do retorno em relação à volatilidade total. |
| **Sortino 20d** | Retorno médio excedente / volatilidade negativa | Mede qualidade do retorno penalizando mais as quedas. |
| **Fator de Qualidade** | `1 + ((0,6 × Sharpe 20d + 0,4 × Sortino 20d) / 2)` limitado entre `0,25x` e `2,00x` | Aumenta ou reduz o Score conforme a qualidade do movimento. |
| **Score Elite** | `Score Simples × Fator de Qualidade` | Score final do ranking, combinando força relativa com qualidade de retorno. |
| **Score Curto Elite** | `Score Curto 5/20 × Fator de Qualidade` | Versão tática ajustada por Sharpe + Sortino. |

**Score simples:**

`Score Simples = média ponderada dos retornos relativos disponíveis`

**Score ELITE:**

`Score Elite = Score Simples × Fator de Qualidade`

**Score Curto 5/20:**

`Score Curto = 0,3 × Relativo 5d + 0,7 × Relativo 20d`

**Score Curto Elite:**

`Score Curto Elite = Score Curto × Fator de Qualidade`

**Fator de Qualidade:** calculado com 60% de Sharpe 20d e 40% de Sortino 20d. O fator é limitado entre 0,25 e 2,00 para evitar distorções por outliers.

Pesos usados no projeto:

| Janela | Peso |
|---:|---:|
| 5 pregões | 10% |
| 20 pregões | 30% |
| 60 pregões | 30% |
| 120 pregões | 30% |
| 252 pregões | 10% |

Quando nem todas as janelas estão selecionadas ou disponíveis, o app recalibra o Score usando apenas os pesos das janelas calculadas.

**Leitura prática:**

- **Score Simples positivo:** ativo está performando melhor que o benchmark no conjunto das janelas.
- **Score Elite:** mantém a força relativa, mas dá mais peso aos ativos com melhor relação retorno/risco.
- **Sharpe 20d:** mede retorno médio por volatilidade total.
- **Sortino 20d:** mede retorno médio por volatilidade negativa (quedas).
- **Score negativo:** ativo está performando pior que o benchmark.
- **Score alto e consistente:** possível liderança relativa.
- **Score caindo ou abaixo da MM20:** perda de força relativa.
- **Score Curto positivo com Score Elite positivo:** timing favorável dentro de liderança.
- **Score Curto negativo com Score Elite positivo:** possível pullback ou perda tática de força.
- **Score Curto positivo com Score Elite negativo:** possível reversão inicial, ainda sem confirmação estrutural.
            """
        )
        if context == "indicator":
            st.info(
                "Na aba Indicador Score, o mesmo cálculo é feito historicamente em cada data, "
                "permitindo visualizar a evolução do Score, a MM20 do Score e o histograma de força/perda de força."
            )


def render_regime_explanation():
    """Mostra os critérios usados para definir o regime de força relativa."""
    with st.expander("Critérios para definição do Regime", expanded=False):
        st.markdown(
            """
O **Regime** resume a condição de força relativa do ativo/índice contra o benchmark selecionado.

O app usa principalmente três informações:

1. **Relativo 20d %**: retorno do ativo em 20 pregões menos o retorno do benchmark no mesmo período.
2. **Relativo 60d %**: retorno do ativo em 60 pregões menos o retorno do benchmark no mesmo período.
3. **RS x MM20 %**: distância da linha de força relativa `Ativo / IBOV` em relação à sua média móvel de 20 períodos.

### Opções de regime

| Regime | Critério usado | Interpretação prática |
|---|---|---|
| **Liderança relativa** | Relativo 20d > 0, Relativo 60d > 0 e RS acima da MM20 | Força relativa sustentada contra o IBOV. |
| **Virando para cima** | Relativo 20d > 0, Relativo 60d < 0 e RS acima da MM20 | Possível início de rotação positiva. |
| **Perdendo força** | Relativo 20d < 0 e Relativo 60d > 0 | Ainda tem desempenho médio positivo, mas perdeu tração recente. |
| **Underperform** | Relativo 20d < 0, Relativo 60d < 0 e RS abaixo da MM20 | Pior desempenho relativo e sem recuperação confirmada. |
| **Neutro** | Critérios mistos ou dados insuficientes | Exige leitura complementar pelo gráfico, preço, volume e fluxo. |

### Como usar

- Para compras, priorize **Liderança relativa** ou **Virando para cima**, desde que o gráfico de preço confirme.
- Para alerta de realização ou perda de tração, observe **Perdendo força**.
- Para evitar compras direcionais, filtre ativos em **Underperform**.
            """
        )


def build_multi_benchmark_summary(asset_prices: pd.DataFrame, benchmark_prices: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Cria resumo multi-benchmark usando o mesmo cálculo de Score Elite para cada benchmark fixo."""
    rows = []
    if asset_prices.empty or benchmark_prices.empty:
        return pd.DataFrame()

    for label, bench_ticker in MULTI_BENCHMARK_OPTIONS.items():
        if bench_ticker not in benchmark_prices.columns:
            continue
        combined = pd.concat([asset_prices, benchmark_prices[[bench_ticker]]], axis=1).dropna(how="all").ffill()
        try:
            rank, _ = calc_metrics(combined, bench_ticker, windows)
        except Exception:
            continue
        if rank.empty:
            continue
        for _, row in rank.iterrows():
            rows.append({
                "Ativo": row.get("Ativo"),
                "Benchmark": label,
                "Ticker Benchmark": bench_ticker,
                "Regime": row.get("Regime"),
                "Score Elite": row.get("Score Elite"),
                "Score Simples": row.get("Score Simples"),
                "Score Curto Elite": row.get("Score Curto Elite"),
                "Sinal Curto": row.get("Sinal Curto"),
                "Sharpe 20d": row.get("Sharpe 20d"),
                "Sortino 20d": row.get("Sortino 20d"),
                "Relativo 20d %": row.get("Relativo 20d %"),
                "Relativo 60d %": row.get("Relativo 60d %"),
                "RS x MM20 %": row.get("RS x MM20 %"),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    pivot = out.pivot_table(index="Ativo", columns="Benchmark", values="Score Elite", aggfunc="first")
    summary = pivot.reset_index()
    bench_cols = [c for c in MULTI_BENCHMARK_OPTIONS.keys() if c in summary.columns]
    summary["Score Multi-Benchmark"] = summary[bench_cols].mean(axis=1, skipna=True)
    summary["Benchmarks positivos"] = (summary[bench_cols] > 0).sum(axis=1)
    summary["Total benchmarks"] = summary[bench_cols].notna().sum(axis=1)
    summary["Consistência"] = summary["Benchmarks positivos"].astype(str) + "/" + summary["Total benchmarks"].astype(str)

    def classify_multi(row):
        score = row.get("Score Multi-Benchmark", np.nan)
        positives = row.get("Benchmarks positivos", 0)
        total = row.get("Total benchmarks", 0)
        if pd.isna(score) or total == 0:
            return "Neutro"
        if score > 0 and positives >= max(3, total - 1):
            return "Liderança relativa"
        if score > 0 and positives >= 2:
            return "Virando para cima"
        if score < 0 and positives <= 1:
            return "Underperform"
        if score < 0 and positives >= 2:
            return "Perdendo força"
        return "Neutro"

    summary["Regime"] = summary.apply(classify_multi, axis=1)
    ordered = ["Ativo", "Regime", "Score Multi-Benchmark", "Consistência"] + bench_cols
    return summary[ordered].sort_values("Score Multi-Benchmark", ascending=False)


def render_table(df: pd.DataFrame, title: str, show_score_explanation: bool = False, show_regime_explanation: bool = False, show_pfr_explanation: bool = False):
    st.subheader(title)
    if show_score_explanation:
        render_score_explanation(context="ranking")
    if show_regime_explanation:
        render_regime_explanation()
    if show_pfr_explanation:
        render_loss_of_strength_explanation()
    if df.empty:
        st.warning("Sem dados suficientes para montar a tabela.")
        return
    df = reorder_ranking_columns(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    styled = df.style.format({c: "{:.2f}" for c in numeric_cols})
    st.dataframe(styled, width="stretch", height=520)

st.title("Força Relativa B3 x Benchmarks")
st.caption("Ranking de ativos e índices setoriais por performance relativa contra IBOV, SPX, NASDAQ, DXY ou benchmark personalizado.")

with st.sidebar:
    st.header("Configuração")
    source_mode = st.radio("Universo de ativos", ["Carteira IBOV automática B3", "Lista manual", "Upload CSV"], index=0)

    st.subheader("Benchmark")
    benchmark_choice = st.selectbox(
        "Benchmark padrão",
        list(BENCHMARK_OPTIONS.keys()),
        index=0,
        help="Selecione um benchmark padrão. O app usará automaticamente o ticker correspondente no Yahoo Finance.",
    )
    custom_benchmark = st.text_input(
        "Benchmark personalizado opcional",
        value="",
        placeholder="Ex.: ^RUT, SPY, EWZ, BOVA11.SA",
        help="Preencha somente se quiser substituir o benchmark padrão selecionado acima.",
    )
    benchmark_input = custom_benchmark.strip() or BENCHMARK_OPTIONS[benchmark_choice]
    st.caption(f"Benchmark em uso: **{benchmark_choice if not custom_benchmark.strip() else 'Personalizado'}** → `{benchmark_input}`")

    start = st.date_input("Data inicial", value=date.today() - timedelta(days=370))
    end = st.date_input("Data final", value=date.today())
    windows_input = st.multiselect("Janelas de ranking", [5, 20, 60, 120, 252], default=[5, 20, 60, 120])
    top_n = st.slider("Quantidade no gráfico", min_value=5, max_value=40, value=15)
    manual_text = st.text_area("Lista manual de ativos", value=DEFAULT_MANUAL_ASSETS)

    uploaded = None
    if source_mode == "Upload CSV":
        uploaded = st.file_uploader("CSV com coluna ticker", type=["csv"])

    st.markdown("---")
    st.subheader("Índices setoriais")
    st.caption("A análise setorial usa somente: IFNC, IMAT, ICON, UTIL e IMOB.")
    allow_sector_edit = st.checkbox("Permitir edição manual dos tickers setoriais", value=False)
    if allow_sector_edit:
        sector_text = st.text_area(
            "Tickers dos índices setoriais no Yahoo",
            value="\n".join([f"{name},{ticker}" for name, ticker in SECTOR_INDICES.items()]),
            help="Formato: Nome do índice,ticker Yahoo. Mantenha somente os cinco índices definidos: IFNC, IMAT, ICON, UTIL e IMOB.",
            height=150,
        )
    else:
        sector_text = "\n".join([f"{name},{ticker}" for name, ticker in SECTOR_INDICES.items()])
        st.dataframe(pd.DataFrame([{"Índice": name, "Ticker Yahoo": ticker} for name, ticker in SECTOR_INDICES.items()]), width="stretch", hide_index=True)

    run = st.button("Atualizar análise", type="primary")


def resolve_tickers(source_mode: str, manual_text: str, uploaded_file) -> List[str]:
    if source_mode == "Carteira IBOV automática B3":
        try:
            ibov_df = fetch_b3_index_portfolio("IBOV")
            tickers = ibov_df["ticker"].tolist() if not ibov_df.empty else FALLBACK_IBOV
            if ibov_df.empty:
                st.warning("Não consegui ler a carteira da B3. Usei a lista fallback editável no código.")
            return tickers
        except Exception as e:
            st.warning(f"Falha ao buscar carteira automática da B3: {e}. Usei fallback local.")
            return FALLBACK_IBOV
    if source_mode == "Lista manual":
        return parse_manual_list(manual_text)
    if uploaded_file is None:
        st.warning("Envie um CSV com a coluna ticker e clique em Atualizar análise.")
        return []
    csv = pd.read_csv(uploaded_file)
    col = "ticker" if "ticker" in csv.columns else csv.columns[0]
    return csv[col].astype(str).str.upper().tolist()


def parse_sector_tickers(sector_text: str) -> Dict[str, str]:
    allowed_sector_codes = {"IFNC", "IMAT", "ICON", "UTIL", "IMOB"}
    sector_tickers = {}
    for line in sector_text.splitlines():
        if not line.strip() or "," not in line:
            continue
        name, tk = line.split(",", 1)
        code = name.strip().split(" - ")[0].upper()
        if code in allowed_sector_codes:
            sector_tickers[name.strip()] = tk.strip()
    return sector_tickers


if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False

if run:
    try:
        benchmark = benchmark_input.strip()
        benchmark_label = benchmark_choice if not custom_benchmark.strip() else f"Personalizado ({benchmark})"
        windows = windows_input
        tickers_br = resolve_tickers(source_mode, manual_text, uploaded)
        if not tickers_br:
            st.stop()

        asset_yahoo = br_to_yahoo(tickers_br)
        all_asset_tickers = tuple(sorted(set(asset_yahoo + [benchmark])))
        prices, used_tickers, failed_tickers = download_prices(all_asset_tickers, start, end)
        ranking, rs_curves = calc_metrics(prices, benchmark, windows)

        fixed_benchmark_tickers = tuple(sorted(set(MULTI_BENCHMARK_OPTIONS.values())))
        multi_benchmark_prices, multi_benchmark_used, multi_benchmark_failed = download_prices(fixed_benchmark_tickers, start, end)
        asset_only_prices = prices[[c for c in prices.columns if c in asset_yahoo]].copy() if not prices.empty else pd.DataFrame()
        multi_benchmark_summary = build_multi_benchmark_summary(asset_only_prices, multi_benchmark_prices, windows)

        sector_tickers = parse_sector_tickers(sector_text)
        sector_all = tuple(sorted(set(list(sector_tickers.values()) + [benchmark])))
        sector_prices, sector_used_tickers, sector_failed_tickers = download_prices(sector_all, start, end)
        sector_ranking, sector_rs = calc_metrics(sector_prices, benchmark, windows)
        if not sector_ranking.empty:
            inverse = {yahoo_to_br(v): k for k, v in sector_tickers.items()}
            sector_ranking["Índice"] = sector_ranking["Ativo"].map(inverse).fillna(sector_ranking["Ativo"])

        st.session_state.analysis_data = {
            "prices": prices,
            "ranking": ranking,
            "rs_curves": rs_curves,
            "benchmark": benchmark,
            "benchmark_label": benchmark_label,
            "windows": windows,
            "failed_tickers": failed_tickers,
            "sector_prices": sector_prices,
            "sector_ranking": sector_ranking,
            "sector_rs": sector_rs,
            "sector_used_tickers": sector_used_tickers,
            "sector_failed_tickers": sector_failed_tickers,
            "sector_tickers": sector_tickers,
            "multi_benchmark_prices": multi_benchmark_prices,
            "multi_benchmark_used": multi_benchmark_used,
            "multi_benchmark_failed": multi_benchmark_failed,
            "multi_benchmark_summary": multi_benchmark_summary,
        }
        st.session_state.analysis_ready = True
        st.success("Dados atualizados e armazenados na sessão. Agora você pode trocar o ativo no indicador sem recalcular.")
    except Exception as e:
        st.error(f"Erro ao atualizar a análise: {e}")
        st.exception(e)
        st.stop()

if not st.session_state.analysis_ready:
    st.info("Configure os parâmetros na lateral e clique em Atualizar análise.")
    st.stop()

try:
    data = st.session_state.analysis_data
    prices = data["prices"]
    ranking = data["ranking"]
    rs_curves = data["rs_curves"]
    benchmark = data["benchmark"]
    benchmark_label = data.get("benchmark_label", benchmark)
    windows = data["windows"]
    failed_tickers = data["failed_tickers"]
    sector_ranking = data["sector_ranking"]
    sector_used_tickers = data["sector_used_tickers"]
    sector_failed_tickers = data["sector_failed_tickers"]
    multi_benchmark_summary = data.get("multi_benchmark_summary", pd.DataFrame())
    multi_benchmark_failed = data.get("multi_benchmark_failed", [])

    if failed_tickers:
        st.warning("Alguns tickers não retornaram dados no Yahoo Finance e foram ignorados: " + ", ".join(failed_tickers))

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Ranking ativos", "Setores", "Linha RS", "Indicador Score", "Mapa de calor", "Multi-benchmark"])

    with tab1:
        render_table(ranking, f"Ranking de ativos contra {benchmark_label}", show_score_explanation=True, show_regime_explanation=True, show_pfr_explanation=True)
        if not ranking.empty:
            fig = px.bar(
                ranking.head(top_n),
                x="Ativo",
                y="Score",
                color="Regime",
                color_discrete_map=REGIME_COLOR_MAP,
                title=f"Top ativos por Score relativo contra {benchmark_label}",
            )
            st.plotly_chart(fig, width="stretch")

    with tab2:
        if sector_used_tickers:
            used_df = pd.DataFrame([{"Ticker solicitado": k, "Ticker usado": v} for k, v in sector_used_tickers.items() if k != benchmark])
            if not used_df.empty:
                st.caption("Tickers efetivamente usados na análise setorial:")
                st.dataframe(used_df, width="stretch", hide_index=True)
        if sector_failed_tickers:
            missing = [t for t in sector_failed_tickers if t != benchmark]
            if missing:
                st.warning("Sem dados para: " + ", ".join(missing) + ". Esses índices foram ignorados no ranking setorial.")
        if not sector_ranking.empty:
            cols = ["Índice"] + [c for c in sector_ranking.columns if c != "Índice"]
            render_table(sector_ranking[cols], f"Ranking setorial contra {benchmark_label}", show_score_explanation=True, show_regime_explanation=True, show_pfr_explanation=True)
            fig2 = px.bar(
                sector_ranking,
                x="Índice",
                y="Score",
                color="Regime",
                color_discrete_map=REGIME_COLOR_MAP,
                title="Setores/índices com maior força relativa",
            )
            st.plotly_chart(fig2, width="stretch")
        else:
            st.warning("Não consegui baixar dados suficientes para os índices setoriais informados. Ajuste os tickers na lateral e clique em Atualizar análise.")

    with tab3:
        if ranking.empty or rs_curves.empty:
            st.warning("Sem curvas de força relativa.")
        else:
            options = ranking["Ativo"].head(30).tolist()
            selected = st.multiselect("Ativos para comparar", options, default=options[: min(5, len(options))], key="rs_line_assets")
            fig3 = go.Figure()
            for br in selected:
                ytk = f"{br}.SA"
                if ytk in rs_curves.columns:
                    fig3.add_trace(go.Scatter(x=rs_curves.index, y=rs_curves[ytk], mode="lines", name=br))
            fig3.update_layout(
                title=f"Linha de Força Relativa normalizada — Ativo / {benchmark_label}, base 100",
                yaxis_title="RS base 100",
                xaxis_title="Data",
                hovermode="x unified",
                height=620,
            )
            enable_horizontal_zoom(fig3, range_slider=True)
            plotly_time_chart(fig3, key="rs_line_chart")
            st.caption(
                "Use os botões 1m/3m/6m/1a/Tudo, arraste o range slider inferior ou use a roda do mouse para ajustar o zoom no eixo horizontal."
            )

    with tab4:
        if ranking.empty or prices.empty:
            st.warning("Sem dados para o indicador visual de Score.")
        else:
            all_available = [yahoo_to_br(c) for c in prices.columns if c != benchmark]
            ranked_assets = ranking["Ativo"].tolist()
            score_options = [a for a in ranked_assets if a in all_available]
            score_options += [a for a in all_available if a not in score_options]
            render_score_explanation(context="indicator")
            render_regime_explanation()
            render_loss_of_strength_explanation()
            selected_score_asset = st.selectbox(
                "Ativo para o indicador visual",
                score_options,
                index=0,
                key="selected_score_asset",
            )
            selected_col = f"{selected_score_asset}.SA"
            if selected_col not in prices.columns and selected_score_asset in prices.columns:
                selected_col = selected_score_asset
            render_score_indicator(
                prices,
                selected_col,
                benchmark,
                windows,
                title=f"Indicador visual de Score Relativo — {selected_score_asset} x {benchmark_label}",
            )
            st.caption(
                "Use os botões 1m/3m/6m/1a/Tudo, arraste o range slider inferior ou use a roda do mouse para ajustar o zoom no eixo horizontal."
            )

    with tab5:
        if ranking.empty:
            st.warning("Sem dados para mapa de calor.")
        else:
            rel_cols = [c for c in ranking.columns if c.startswith("Relativo")]
            heat = ranking.set_index("Ativo")[rel_cols].head(50)
            fig4 = px.imshow(heat, aspect="auto", text_auto=".1f", title=f"Mapa de calor: retorno relativo contra {benchmark_label} (%)")
            st.plotly_chart(fig4, width="stretch")


    with tab6:
        st.subheader("Radar multi-benchmark")
        st.markdown(
            """
Este painel compara os mesmos ativos simultaneamente contra os benchmarks fixos: **IBOV**, **SPX**, **NASDAQ** e **DXY**.

A coluna **Score Multi-Benchmark** é a média dos Scores Elite disponíveis contra cada benchmark.
A coluna **Consistência** mostra em quantos benchmarks o ativo está com Score Elite positivo.
            """
        )
        if multi_benchmark_failed:
            st.warning("Alguns benchmarks fixos não retornaram dados e foram ignorados: " + ", ".join(multi_benchmark_failed))
        if multi_benchmark_summary.empty:
            st.warning("Sem dados suficientes para montar o radar multi-benchmark.")
        else:
            render_table(multi_benchmark_summary, "Ranking multi-benchmark", show_score_explanation=False, show_regime_explanation=True)
            mb_top = multi_benchmark_summary.head(top_n)
            fig_mb = px.bar(
                mb_top,
                x="Ativo",
                y="Score Multi-Benchmark",
                color="Regime",
                color_discrete_map=REGIME_COLOR_MAP,
                title="Top ativos por Score Multi-Benchmark",
            )
            st.plotly_chart(fig_mb, width="stretch")

            heat_cols = [c for c in MULTI_BENCHMARK_OPTIONS.keys() if c in multi_benchmark_summary.columns]
            if heat_cols:
                heat_mb = multi_benchmark_summary.set_index("Ativo")[heat_cols].head(50)
                fig_mb_heat = px.imshow(
                    heat_mb,
                    aspect="auto",
                    text_auto=".1f",
                    title="Mapa de calor multi-benchmark — Score Elite por benchmark",
                )
                st.plotly_chart(fig_mb_heat, width="stretch")

    st.download_button(
        "Baixar ranking em CSV",
        data=ranking.to_csv(index=False).encode("utf-8-sig"),
        file_name="ranking_forca_relativa_b3.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error(f"Erro na execução: {e}")
    st.exception(e)
