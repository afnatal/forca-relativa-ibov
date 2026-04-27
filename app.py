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

st.set_page_config(page_title="Força Relativa B3 x IBOV", layout="wide")

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
}

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
        weights = {5: 0.20, 20: 0.35, 60: 0.30, 120: 0.15, 252: 0.10}
        used_weight = 0.0
        for w in windows:
            if len(asset) > w:
                ret_asset = asset.iloc[-1] / asset.iloc[-w - 1] - 1
                ret_bench = bm.iloc[-1] / bm.iloc[-w - 1] - 1
                rel = ret_asset - ret_bench
                row[f"Ativo {w}d %"] = ret_asset * 100
                row[f"IBOV {w}d %"] = ret_bench * 100
                row[f"Relativo {w}d %"] = rel * 100
                wt = weights.get(w, 1 / len(windows))
                score += rel * wt
                used_weight += wt
        rs_ma = rs.rolling(ma_window).mean()
        row["RS Atual"] = rs_norm.iloc[-1]
        row["RS x MM20 %"] = ((rs.iloc[-1] / rs_ma.iloc[-1] - 1) * 100) if len(rs_ma.dropna()) else np.nan
        row["Score"] = (score / used_weight * 100) if used_weight else np.nan
        row["Regime"] = classify_regime(row)
        rows.append(row)
    ranking = pd.DataFrame(rows).sort_values("Score", ascending=False) if rows else pd.DataFrame()
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
    weights = {5: 0.20, 20: 0.35, 60: 0.30, 120: 0.15, 252: 0.10}
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
    df = pd.DataFrame(index=aligned.index)
    df["Score"] = score_pct
    df["MM20 Score"] = df["Score"].rolling(ma_window).mean()
    df["Histograma"] = df["Score"] - df["MM20 Score"]
    df["Direção"] = df["Score"].diff()

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

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=score_df.index,
        y=score_df["Score"],
        name="Histograma Score",
        marker_color=bar_colors,
        opacity=0.55,
        hovertemplate="Data=%{x}<br>Score=%{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=score_df.index,
        y=score_df["Score"],
        mode="lines",
        name="Score Relativo",
        line=dict(width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=score_df.index,
        y=score_df["MM20 Score"],
        mode="lines",
        name="MM20 do Score",
        line=dict(width=2, dash="dot"),
    ))
    fig.add_hline(y=0, line_width=1, line_dash="dash", annotation_text="Zero", annotation_position="bottom right")
    fig.update_layout(
        title=title,
        yaxis_title="Score relativo contra IBOV (%)",
        xaxis_title="Data",
        legend_title="Indicador",
        hovermode="x unified",
        height=560,
    )
    st.plotly_chart(fig, width="stretch")

    last = score_df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score atual", f"{last['Score']:.2f}%")
    c2.metric("MM20 Score", f"{last['MM20 Score']:.2f}%" if pd.notna(last["MM20 Score"]) else "n/d")
    c3.metric(
        "Distância Score x MM20",
        f"{(last['Score'] - last['MM20 Score']):.2f} p.p." if pd.notna(last["MM20 Score"]) else "n/d",
    )
    c4.metric("Condição", str(last["Condição"]))

    st.caption(
        "Leitura: Score acima de zero indica outperform contra o IBOV; "
        "Score acima da MM20 indica liderança relativa ganhando consistência; "
        "as barras mostram a direção do score."
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


def parse_manual_list(text: str) -> List[str]:
    tokens = text.replace(";", ",").replace("\n", ",").split(",")
    return [t.strip().upper() for t in tokens if t.strip()]


def render_score_explanation(context: str = "ranking"):
    """Mostra uma explicação curta e consistente sobre o cálculo do Score."""
    with st.expander("Como o Score é calculado", expanded=False):
        st.markdown(
            """
O **Score** é uma média ponderada da performance relativa do ativo contra o IBOV.

**Performance relativa de cada janela:**

`Relativo Nd % = Retorno do ativo em N pregões - Retorno do IBOV em N pregões`

**Score final:**

`Score = média ponderada dos retornos relativos disponíveis`

Pesos usados no projeto:

| Janela | Peso |
|---:|---:|
| 5 pregões | 20% |
| 20 pregões | 35% |
| 60 pregões | 30% |
| 120 pregões | 15% |
| 252 pregões | 10% |

Quando nem todas as janelas estão selecionadas ou disponíveis, o app recalibra o Score usando apenas os pesos das janelas calculadas.

**Leitura prática:**

- **Score positivo:** ativo está performando melhor que o IBOV no conjunto das janelas.
- **Score negativo:** ativo está performando pior que o IBOV.
- **Score alto e consistente:** possível liderança relativa.
- **Score caindo ou abaixo da MM20:** perda de força relativa.
            """
        )
        if context == "indicator":
            st.info(
                "Na aba Indicador Score, o mesmo cálculo é feito historicamente em cada data, "
                "permitindo visualizar a evolução do Score, a MM20 do Score e o histograma de força/perda de força."
            )


def render_table(df: pd.DataFrame, title: str, show_score_explanation: bool = False):
    st.subheader(title)
    if show_score_explanation:
        render_score_explanation(context="ranking")
    if df.empty:
        st.warning("Sem dados suficientes para montar a tabela.")
        return
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    styled = df.style.format({c: "{:.2f}" for c in numeric_cols})
    st.dataframe(styled, width="stretch", height=520)

st.title("Força Relativa B3 x IBOV")
st.caption("Ranking de ativos e índices setoriais por performance relativa contra o Ibovespa.")

with st.sidebar:
    st.header("Configuração")
    source_mode = st.radio("Universo de ativos", ["Carteira IBOV automática B3", "Lista manual", "Upload CSV"], index=0)
    benchmark_input = st.text_input("Benchmark", value="^BVSP", help="Yahoo Finance: ^BVSP para Ibovespa.")
    start = st.date_input("Data inicial", value=date.today() - timedelta(days=370))
    end = st.date_input("Data final", value=date.today())
    windows_input = st.multiselect("Janelas de ranking", [5, 20, 60, 120, 252], default=[5, 20, 60, 120])
    top_n = st.slider("Quantidade no gráfico", min_value=5, max_value=40, value=15)
    manual_text = st.text_area("Lista manual de ativos", value="PETR4, VALE3, ITUB4, BBAS3, BBDC4, BPAC11, PRIO3")

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
        windows = windows_input
        tickers_br = resolve_tickers(source_mode, manual_text, uploaded)
        if not tickers_br:
            st.stop()

        asset_yahoo = br_to_yahoo(tickers_br)
        all_asset_tickers = tuple(sorted(set(asset_yahoo + [benchmark])))
        prices, used_tickers, failed_tickers = download_prices(all_asset_tickers, start, end)
        ranking, rs_curves = calc_metrics(prices, benchmark, windows)

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
            "windows": windows,
            "failed_tickers": failed_tickers,
            "sector_prices": sector_prices,
            "sector_ranking": sector_ranking,
            "sector_rs": sector_rs,
            "sector_used_tickers": sector_used_tickers,
            "sector_failed_tickers": sector_failed_tickers,
            "sector_tickers": sector_tickers,
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
    windows = data["windows"]
    failed_tickers = data["failed_tickers"]
    sector_ranking = data["sector_ranking"]
    sector_used_tickers = data["sector_used_tickers"]
    sector_failed_tickers = data["sector_failed_tickers"]

    if failed_tickers:
        st.warning("Alguns tickers não retornaram dados no Yahoo Finance e foram ignorados: " + ", ".join(failed_tickers))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ranking ativos", "Setores", "Linha RS", "Indicador Score", "Mapa de calor"])

    with tab1:
        render_table(ranking, "Ranking de ativos contra o IBOV", show_score_explanation=True)
        if not ranking.empty:
            fig = px.bar(ranking.head(top_n), x="Ativo", y="Score", color="Regime", title="Top ativos por Score relativo")
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
            render_table(sector_ranking[cols], "Ranking setorial contra o IBOV", show_score_explanation=True)
            fig2 = px.bar(sector_ranking, x="Índice", y="Score", color="Regime", title="Setores/índices com maior força relativa")
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
            fig3.update_layout(title="Linha de Força Relativa normalizada — Ativo / IBOV, base 100", yaxis_title="RS base 100")
            st.plotly_chart(fig3, width="stretch")

    with tab4:
        if ranking.empty or prices.empty:
            st.warning("Sem dados para o indicador visual de Score.")
        else:
            all_available = [yahoo_to_br(c) for c in prices.columns if c != benchmark]
            ranked_assets = ranking["Ativo"].tolist()
            score_options = [a for a in ranked_assets if a in all_available]
            score_options += [a for a in all_available if a not in score_options]
            render_score_explanation(context="indicator")
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
                title=f"Indicador visual de Score Relativo — {selected_score_asset} x IBOV",
            )

    with tab5:
        if ranking.empty:
            st.warning("Sem dados para mapa de calor.")
        else:
            rel_cols = [c for c in ranking.columns if c.startswith("Relativo")]
            heat = ranking.set_index("Ativo")[rel_cols].head(50)
            fig4 = px.imshow(heat, aspect="auto", text_auto=".1f", title="Mapa de calor: retorno relativo contra IBOV (%)")
            st.plotly_chart(fig4, width="stretch")

    st.download_button(
        "Baixar ranking em CSV",
        data=ranking.to_csv(index=False).encode("utf-8-sig"),
        file_name="ranking_forca_relativa_b3.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error(f"Erro na execução: {e}")
    st.exception(e)
