import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")

st.title("RAG Evaluation Comparison Dashboard")
st.caption("DailyMed Information Retrieval System - UC3M NLP Project 2026")

@st.cache_data
def load_data():
    return pd.read_csv("evaluation/grid_results/grid_retrieval_comparison.csv")

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
k_vals = st.sidebar.multiselect(
    "Select top-k values:", 
    [3, 5, 8, 10], 
    default=[3, 5, 8, 10]
)

t_options = [0.20, 0.30, 0.40, 0.60, 0.80]
selected_t = st.sidebar.select_slider(
    "Select Minimum Threshold:",
    options=t_options,
    value=0.20
)

# Filter dataframe based on controls
filtered_df = df[df['k'].isin(k_vals) & (df['threshold'] >= selected_t)]

# Display main metrics table
st.subheader("General Comparison Table")
st.dataframe(
    filtered_df[['k', 'threshold', 'hit@k', 'recall@k', 'precision@k', 'mrr', 'lat_avg_ms', 'lat_p95_ms']], 
    use_container_width=True
)

# Create a two-column layout for the charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Recall@K")
    chart_rec = alt.Chart(filtered_df).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('threshold:O', title='Threshold'),
        y=alt.Y('recall@k:Q', title='Recall@K', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('k:N', title='top-k'),
        tooltip=['k', 'threshold', 'recall@k']
    ).properties(height=300)
    st.altair_chart(chart_rec, use_container_width=True)

with col2:
    st.subheader("Precision@K")
    chart_prec = alt.Chart(filtered_df).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('threshold:O', title='Threshold'),
        y=alt.Y('precision@k:Q', title='Precision@K', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('k:N', title='top-k'),
        tooltip=['k', 'threshold', 'precision@k']
    ).properties(height=300)
    st.altair_chart(chart_prec, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("MRR")
    chart_mrr = alt.Chart(filtered_df).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('threshold:O', title='Threshold'),
        y=alt.Y('mrr:Q', title='MRR', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('k:N', title='top-k'),
        tooltip=['k', 'threshold', 'mrr']
    ).properties(height=300)
    st.altair_chart(chart_mrr, use_container_width=True)

with col4:
    st.subheader("Average Latency (ms)")
    chart_lat = alt.Chart(filtered_df).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('threshold:O', title='Threshold'),
        y=alt.Y('lat_avg_ms:Q', title='Average Latency (ms)'),
        color=alt.Color('k:N', title='top-k'),
        tooltip=['k', 'threshold', 'lat_avg_ms']
    ).properties(height=300)
    st.altair_chart(chart_lat, use_container_width=True)

# Download button
st.download_button(
    label="Download results as CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name='final_evaluation.csv',
    mime='text/csv'
)