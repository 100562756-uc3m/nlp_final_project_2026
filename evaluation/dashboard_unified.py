# streamlit run dashboard_unified.py
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

st.set_page_config(
    page_title="DailyMed RAG — Evaluation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0f1117; }
    [data-testid="stSidebar"] { background-color: #1a1d27; border-right: 1px solid #2d2f3e; }
    .metric-card {
        background: #1e2130;
        border: 1px solid #2d3048;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        text-align: center;
    }
    .metric-label { font-size: 11px; color: #8b8fa8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #e8eaf6; }
    .metric-sub   { font-size: 11px; color: #5c6080; margin-top: 2px; }
    .section-header {
        font-size: 11px; font-weight: 600; letter-spacing: 0.12em;
        text-transform: uppercase; color: #6c6f8a;
        border-bottom: 1px solid #2d2f3e; padding-bottom: 6px; margin: 1.5rem 0 1rem;
    }
    .badge-good { background: #1b3a2e; color: #4ade80; border-radius: 6px; padding: 2px 10px; font-size: 11px; font-weight: 600; }
    .badge-mid  { background: #3a2e10; color: #facc15; border-radius: 6px; padding: 2px 10px; font-size: 11px; font-weight: 600; }
    .badge-low  { background: #3a1b1b; color: #f87171; border-radius: 6px; padding: 2px 10px; font-size: 11px; font-weight: 600; }
    .tab-desc { font-size: 13px; color: #6c6f8a; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data
def load_retrieval():
    return pd.read_csv("evaluation/grid_results/grid_retrieval_comparison.csv")

@st.cache_data
def load_generation():
    return pd.read_csv("evaluation/generation_results.csv")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## RAG Evaluation")
    st.markdown("DailyMed · UC3M NLP 2026")
    st.divider()

    page = st.radio(
        "Section",
        ["Overview", "Retrieval", "Generation", "Error Analysis"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Retrieval filters**")
    k_filter = st.multiselect("top-k", [3, 5, 8, 10], default=[3, 5, 8, 10])
    t_min = st.select_slider(
        "Min threshold",
        options=[0.20, 0.30, 0.40, 0.60, 0.80],
        value=0.20
    )

    st.divider()

    try:
        ret_df = load_retrieval()
        st.download_button(
            "Download retrieval CSV",
            ret_df.to_csv(index=False).encode(),
            "retrieval_results.csv", "text/csv"
        )
    except Exception:
        pass
    try:
        gen_df = load_generation()
        st.download_button(
            "Download generation CSV",
            gen_df.to_csv(index=False).encode(),
            "generation_results.csv", "text/csv"
        )
    except Exception:
        pass


# ── Chart theme ───────────────────────────────────────────────────────────────
DARK_THEME = {
    "config": {
        "background": "#1e2130",
        "view": {"stroke": "transparent"},
        "axis": {
            "domainColor": "#2d2f3e", "gridColor": "#252837",
            "labelColor": "#8b8fa8", "titleColor": "#8b8fa8",
            "tickColor": "#2d2f3e"
        },
        "legend": {"labelColor": "#8b8fa8", "titleColor": "#8b8fa8"},
        "title": {"color": "#c5c8e0"}
    }
}

COLOR_SCALE = alt.Scale(
    domain=[3, 5, 8, 10],
    range=["#818cf8", "#34d399", "#fb923c", "#f472b6"]
)


def dark_chart(chart):
    return chart.configure(**DARK_THEME["config"])


# ── Helpers ───────────────────────────────────────────────────────────────────
def metric_card(label, value, sub=""):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>"""


def best_config(df: pd.DataFrame) -> pd.Series:
    """
    Selects best config using a composite score that balances:
    - Recall@K (weight 0.35): primary retrieval quality
    - MRR      (weight 0.30): ranking quality
    - Precision@K (weight 0.20): noise control
    - Latency  (weight 0.15): speed (lower is better, normalised)
    """
    df = df.copy()
    # Normalise latency to [0,1] inverted (lower lat → higher score)
    lat_max = df["lat_avg_ms"].max()
    lat_min = df["lat_avg_ms"].min()
    if lat_max > lat_min:
        df["lat_norm"] = 1.0 - (df["lat_avg_ms"] - lat_min) / (lat_max - lat_min)
    else:
        df["lat_norm"] = 1.0

    df["composite"] = (
        0.35 * df["recall@k"] +
        0.30 * df["mrr"] +
        0.20 * df["precision@k"] +
        0.15 * df["lat_norm"]
    )
    return df.loc[df["composite"].idxmax()]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("RAG Evaluation Dashboard")
    st.markdown('<p class="tab-desc">DailyMed Information Retrieval System — UC3M NLP Project 2026</p>', unsafe_allow_html=True)

    ret_ok = gen_ok = False
    try:
        ret_df = load_retrieval()
        ret_ok = True
    except Exception:
        st.warning("Retrieval results not found. Run `evaluate_rag.py` first.")

    try:
        gen_df = load_generation()
        gen_ok = True
    except Exception:
        st.warning("Generation results not found. Run `evaluate_generation.py` first.")

    if ret_ok:
        # ── Best config using composite score ─────────────────────────────
        best = best_config(ret_df)
        st.markdown('<div class="section-header">Retrieval — best configuration (composite score: recall × 0.35 + MRR × 0.30 + precision × 0.20 + latency × 0.15)</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.markdown(metric_card("Best config", f"k={int(best['k'])}, t={best['threshold']}"), unsafe_allow_html=True)
        c2.markdown(metric_card("Hit@K",        f"{best['hit@k']:.2%}",        "≥1 relevant found"), unsafe_allow_html=True)
        c3.markdown(metric_card("Recall@K",     f"{best['recall@k']:.2%}",     "fraction relevant"), unsafe_allow_html=True)
        c4.markdown(metric_card("Precision@K",  f"{best['precision@k']:.2%}",  "noise ratio"), unsafe_allow_html=True)
        c5.markdown(metric_card("MRR",          f"{best['mrr']:.2%}",          "rank quality"), unsafe_allow_html=True)
        c6.markdown(metric_card("Avg latency",  f"{best['lat_avg_ms']:.0f} ms","retrieval speed"), unsafe_allow_html=True)

    if gen_ok:
        st.markdown('<div class="section-header">Generation quality</div>', unsafe_allow_html=True)
        answerable_gen = gen_df[gen_df["groundedness"] > 0]
        refusal_acc = gen_df["refusal_correct"].mean()
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Avg groundedness", f"{answerable_gen['groundedness'].mean():.2f}/5", "faithfulness to sources"), unsafe_allow_html=True)
        c2.markdown(metric_card("Avg relevance",    f"{answerable_gen['relevance'].mean():.2f}/5",    "answers the question"), unsafe_allow_html=True)
        c3.markdown(metric_card("Refusal accuracy", f"{refusal_acc:.2%}",                             "unsafe Qs refused"), unsafe_allow_html=True)
        c4.markdown(metric_card("Avg latency",      f"{gen_df['latency'].mean():.1f}s",               "full pipeline"), unsafe_allow_html=True)

    if ret_ok and gen_ok:
        st.markdown('<div class="section-header">Pipeline summary</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Recall@K across configurations
            filt = ret_df[ret_df["k"].isin(k_filter) & (ret_df["threshold"] >= t_min)]
            chart = alt.Chart(filt).mark_line(point=True, strokeWidth=2).encode(
                x=alt.X("threshold:O", title="Threshold"),
                y=alt.Y("recall@k:Q", title="Recall@K", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("k:N", scale=COLOR_SCALE, title="top-k"),
                tooltip=["k", "threshold", "recall@k", "mrr"]
            ).properties(title="Recall@K across configurations", height=280)
            st.altair_chart(dark_chart(chart), use_container_width=True)

        with col2:
            # Refusal accuracy donut
            ref_counts = gen_df["refusal_correct"].value_counts().reset_index()
            ref_counts.columns = ["Status", "Count"]
            ref_counts["Status"] = ref_counts["Status"].map({True: "Correct refusal", False: "Wrong response"})
            pie = alt.Chart(ref_counts).mark_arc(innerRadius=55, outerRadius=100).encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color("Status:N", scale=alt.Scale(
                    domain=["Correct refusal", "Wrong response"],
                    range=["#34d399", "#f87171"]
                )),
                tooltip=["Status", "Count"]
            ).properties(title="Refusal accuracy (safety)", height=280)
            st.altair_chart(dark_chart(pie), use_container_width=True)

    elif ret_ok and not gen_ok:
        # Only retrieval available — show Recall@K alone
        st.markdown('<div class="section-header">Retrieval summary</div>', unsafe_allow_html=True)
        filt = ret_df[ret_df["k"].isin(k_filter) & (ret_df["threshold"] >= t_min)]
        chart = alt.Chart(filt).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("threshold:O", title="Threshold"),
            y=alt.Y("recall@k:Q", title="Recall@K", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("k:N", scale=COLOR_SCALE, title="top-k"),
            tooltip=["k", "threshold", "recall@k", "mrr"]
        ).properties(title="Recall@K across configurations", height=300)
        st.altair_chart(dark_chart(chart), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Retrieval":
    st.title("Retrieval Evaluation")
    st.markdown('<p class="tab-desc">Hit@K · Recall@K · Precision@K · MRR — grid search across top-k and similarity threshold</p>', unsafe_allow_html=True)

    try:
        df = load_retrieval()
    except Exception:
        st.error("retrieval CSV not found.")
        st.stop()

    filt = df[df["k"].isin(k_filter) & (df["threshold"] >= t_min)]

    # Best config highlight
    best = best_config(filt if not filt.empty else df)
    st.markdown('<div class="section-header">Best configuration (composite score)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(metric_card("Config",       f"k={int(best['k'])}, t={best['threshold']}"), unsafe_allow_html=True)
    c2.markdown(metric_card("Recall@K",     f"{best['recall@k']:.2%}"), unsafe_allow_html=True)
    c3.markdown(metric_card("Precision@K",  f"{best['precision@k']:.2%}"), unsafe_allow_html=True)
    c4.markdown(metric_card("MRR",          f"{best['mrr']:.2%}"), unsafe_allow_html=True)
    c5.markdown(metric_card("Avg latency",  f"{best['lat_avg_ms']:.0f} ms"), unsafe_allow_html=True)

    st.markdown('<div class="section-header">Metric charts</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    def line_chart(metric, title, col):
        c = alt.Chart(filt).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("threshold:O", title="Threshold"),
            y=alt.Y(f"{metric}:Q", title=title, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("k:N", scale=COLOR_SCALE, title="top-k"),
            tooltip=["k", "threshold", metric]
        ).properties(title=title, height=260)
        col.altair_chart(dark_chart(c), use_container_width=True)

    line_chart("recall@k",    "Recall@K",    col1)
    line_chart("precision@k", "Precision@K", col2)
    col3, col4 = st.columns(2)
    line_chart("mrr", "MRR", col3)

    lat_chart = alt.Chart(filt).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("threshold:O", title="Threshold"),
        y=alt.Y("lat_avg_ms:Q", title="Avg latency (ms)"),
        color=alt.Color("k:N", scale=COLOR_SCALE, title="top-k"),
        tooltip=["k", "threshold", "lat_avg_ms", "lat_p95_ms"]
    ).properties(title="Average latency (ms)", height=260)
    col4.altair_chart(dark_chart(lat_chart), use_container_width=True)

    # Language breakdown
    st.markdown('<div class="section-header">English vs foreign language breakdown</div>', unsafe_allow_html=True)
    lang_melt = filt[["k", "threshold", "hit@k_en", "hit@k_others"]].melt(
        id_vars=["k", "threshold"],
        value_vars=["hit@k_en", "hit@k_others"],
        var_name="Language", value_name="Hit@K"
    )
    lang_melt["Language"] = lang_melt["Language"].map({"hit@k_en": "English", "hit@k_others": "Foreign"})
    lang_melt["config"] = "k=" + lang_melt["k"].astype(str) + " t=" + lang_melt["threshold"].astype(str)

    lang_chart = alt.Chart(lang_melt).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("config:N", title="Configuration", sort=None),
        y=alt.Y("Hit@K:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Language:N", scale=alt.Scale(range=["#818cf8", "#fb923c"])),
        xOffset="Language:N", # <--- ESTO SEPARA LAS BARRAS LADO A LADO
        tooltip=["config", "Language", "Hit@K"]
    ).properties(title="Hit@K: English vs foreign queries", height=280)
    
    st.altair_chart(dark_chart(lang_chart), use_container_width=True)
    
# Summary table
    st.markdown('<div class="section-header">Full grid comparison</div>', unsafe_allow_html=True)
    display = filt[["k", "threshold", "hit@k", "recall@k", "precision@k", "mrr", "lat_avg_ms", "lat_p95_ms"]].copy()
    st.dataframe(
        display.style.highlight_max(
            subset=["hit@k", "recall@k", "mrr"],
            color="#1b3a2e"
        ).format({
            "hit@k": "{:.4f}", "recall@k": "{:.4f}",
            "precision@k": "{:.4f}", "mrr": "{:.4f}",
            "lat_avg_ms": "{:.1f}", "lat_p95_ms": "{:.1f}"
        }),
        use_container_width=True, hide_index=True
    )



# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GENERATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Generation":
    st.title("Generation Evaluation")
    st.markdown('<p class="tab-desc">LLM-as-a-Judge · Groundedness · Relevance · Safety (Refusal accuracy) · Latency</p>', unsafe_allow_html=True)

    try:
        df = load_generation()
    except Exception:
        st.error("generation_results.csv not found. Run evaluate_generation.py first.")
        st.stop()

    answerable = df[df["groundedness"] > 0]
    refusal_acc = df["refusal_correct"].mean()

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Avg groundedness", f"{answerable['groundedness'].mean():.2f}/5", "hallucination check"), unsafe_allow_html=True)
    c2.markdown(metric_card("Avg relevance",    f"{answerable['relevance'].mean():.2f}/5",    "answer quality"), unsafe_allow_html=True)
    c3.markdown(metric_card("Refusal accuracy", f"{refusal_acc:.2%}",                         "safety"), unsafe_allow_html=True)
    c4.markdown(metric_card("Avg latency",      f"{df['latency'].mean():.1f}s",               "full pipeline"), unsafe_allow_html=True)

    # --- VISUALIZATIONS ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Safety: Refusal Accuracy</div>', unsafe_allow_html=True)
        # Prepare data for pie chart
        refusal_counts = df['refusal_correct'].value_counts().reset_index()
        refusal_counts.columns = ['Status', 'Count']
        refusal_counts['Status'] = refusal_counts['Status'].map({True: 'Correct', False: 'Incorrect'})
        
        pie = alt.Chart(refusal_counts).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Status", type="nominal", 
                            scale=alt.Scale(domain=['Correct', 'Incorrect'], 
                                           range=['#34d399', '#f87171'])),
            tooltip=['Status', 'Count']
        ).properties(height=280)
        st.altair_chart(dark_chart(pie), use_container_width=True)
        st.caption("Measures if the bot correctly identified unanswerable/dangerous queries.")

    with col2:
        st.markdown('<div class="section-header">Quality Distribution</div>', unsafe_allow_html=True)
        
        dist_df = df.melt(id_vars=['id'], value_vars=['groundedness', 'relevance'], 
                          var_name='Metric', value_name='Score')
        
        # Gráfico de barras agrupadas (lado a lado) usando xOffset
        hist = alt.Chart(dist_df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X('Score:O', title='Score (1-5)'),
            y=alt.Y('count():Q', title='Number of Samples'), # Quitamos stack=None
            color=alt.Color('Metric:N', scale=alt.Scale(range=['#818cf8', '#34d399'])),
            xOffset='Metric:N', # <--- ESTO PONE LAS BARRAS LADO A LADO
            tooltip=['Metric', 'Score', 'count()']
        ).properties(height=280)
        
        st.altair_chart(dark_chart(hist), use_container_width=True)
        st.caption("Distribution of Groundedness vs Relevance scores.")

    # ── 3. Quality by language ─────────────────────────────────────────────
    if "language" in df.columns:
        st.markdown('<div class="section-header">Quality by language</div>', unsafe_allow_html=True)
        lang_grp = answerable.groupby("language")[["groundedness", "relevance"]].mean().reset_index()
        lang_melt = lang_grp.melt(id_vars="language", var_name="Metric", value_name="Score")
        
        lang_chart = alt.Chart(lang_melt).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("language:N", title="Language"),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 5])),
            color=alt.Color("Metric:N", scale=alt.Scale(range=["#818cf8", "#34d399"])),
            xOffset="Metric:N",
            tooltip=["language", "Metric", alt.Tooltip("Score:Q", format=".2f")]
        ).properties(height=180)
        
        st.altair_chart(dark_chart(lang_chart), use_container_width=True)

    # ── 4. Per-question results table ───────────────────────────────────────
    st.markdown('<div class="section-header">Per-question results</div>', unsafe_allow_html=True)
    st.dataframe(
        df[["id", "question", "language", "expected_refusal",
            "refusal_correct", "groundedness", "relevance", "latency", "reasoning"]],
        use_container_width=True, hide_index=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Error Analysis":
    st.title("Error Analysis")
    st.markdown('<p class="tab-desc">Questions where retrieval or generation failed — diagnose system weaknesses</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Retrieval failures (Hit@K = 0)</div>', unsafe_allow_html=True)
        try:
            import glob
            detail_files = glob.glob("evaluation/grid_results/combinations/k10_t0.20_details.csv")
            if detail_files:
                det = pd.read_csv(detail_files[0])
                failures = det[det["hit"] == 0][["id", "language", "original_question", "expected_support", "retrieved_documents", "recall", "mrr"]]
                st.caption(f"{len(failures)} questions failed retrieval at k=10, t=0.20 (most permissive config)")
                st.dataframe(failures, use_container_width=True, hide_index=True)
            else:
                st.info("Run evaluate_rag.py to generate detail files.")
        except Exception:
            st.info("Detail files not found.")

    with col2:
        st.markdown('<div class="section-header">Generation failures (Groundedness < 4)</div>', unsafe_allow_html=True)
        try:
            gen_df = load_generation()
            gen_errors = gen_df[
                (gen_df["groundedness"] > 0) & (gen_df["groundedness"] < 4)
            ][["id", "question", "language", "groundedness", "relevance", "reasoning"]]
            if gen_errors.empty:
                st.success("No critical hallucinations detected. All answerable questions scored ≥ 4.")
            else:
                st.caption(f"{len(gen_errors)} questions scored below 4 on groundedness")
                st.dataframe(gen_errors, use_container_width=True, hide_index=True)
        except Exception:
            st.info("Run evaluate_generation.py to generate generation results.")

    st.markdown('<div class="section-header">Safety failures — wrong refusal behavior</div>', unsafe_allow_html=True)
    try:
        gen_df = load_generation()
        safety_fails = gen_df[gen_df["refusal_correct"] == False][
            ["id", "question", "language", "expected_refusal", "actual_refusal", "reasoning"]
        ]
        if safety_fails.empty:
            st.success("All refusal decisions were correct.")
        else:
            st.warning(f"{len(safety_fails)} questions had incorrect refusal behavior.")
            st.dataframe(safety_fails, use_container_width=True, hide_index=True)
    except Exception:
        st.info("Generation results not available.")