# streamlit run dashboard_gen.py
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

st.set_page_config(page_title="Generation Evaluation Dashboard", layout="wide")

st.title("RAG Generation Quality Dashboard")
st.caption("Llama 3.1:8b Evaluation - Faithfulness, Relevance, and Safety Metrics")

@st.cache_data
def load_gen_data():
    return pd.read_csv("evaluation/generation_results.csv")

df = load_gen_data()

# --- TOP LEVEL METRICS ---
avg_groundedness = df['groundedness'].mean()
avg_relevance = df['relevance'].mean()
refusal_acc = (df['refusal_correct'].sum() / len(df)) * 100
avg_lat = df['latency'].mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Avg Groundedness", f"{avg_groundedness:.2f} / 5")
m2.metric("Avg Relevance", f"{avg_relevance:.2f} / 5")
m3.metric("Safety (Refusal Acc)", f"{refusal_acc:.1f}%")
m4.metric("Avg Latency", f"{avg_lat:.2f}s")

st.divider()

# --- VISUALIZATIONS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Safety: Refusal Accuracy")
    # Prepare data for pie chart
    refusal_counts = df['refusal_correct'].value_counts().reset_index()
    refusal_counts.columns = ['Status', 'Count']
    refusal_counts['Status'] = refusal_counts['Status'].map({True: 'Correct', False: 'Incorrect'})
    
    pie = alt.Chart(refusal_counts).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Status", type="nominal", 
                        scale=alt.Scale(domain=['Correct', 'Incorrect'], 
                                       range=['#2ca02c', '#d62728'])),
        tooltip=['Status', 'Count']
    ).properties(height=300)
    st.altair_chart(pie, use_container_width=True)
    st.caption("Measures if the bot correctly identified unanswerable/dangerous queries.")

with col2:
    st.subheader("Quality Distribution")
    # Combine Groundedness and Relevance for a distribution plot
    dist_df = df.melt(id_vars=['id'], value_vars=['groundedness', 'relevance'], 
                      var_name='Metric', value_name='Score')
    
    hist = alt.Chart(dist_df).mark_bar(opacity=0.7).encode(
        x=alt.X('Score:O', title='Score (1-5)'),
        y=alt.Y('count():Q', stack=None, title='Number of Samples'),
        color=alt.Color('Metric:N', scale=alt.Scale(range=['#1f77b4', '#ff7f0e'])),
        column='Metric:N'
    ).properties(width=200, height=300)
    st.altair_chart(hist)

st.divider()

# --- LANGUAGE ANALYSIS ---
st.subheader("Quality by Language")
lang_metrics = df.groupby('language')[['groundedness', 'relevance']].mean().reset_index()

lang_chart = alt.Chart(lang_metrics).mark_bar().encode(
    x=alt.X('language:N', title='Language Code'),
    y=alt.Y('groundedness:Q', title='Avg Groundedness', scale=alt.Scale(domain=[0, 5])),
    color=alt.value("#4c78a8"),
    tooltip=['language', 'groundedness', 'relevance']
).properties(height=300)

st.altair_chart(lang_chart, use_container_width=True)

# --- ERROR DEEP DIVE ---
st.divider()
st.subheader("Error Analysis (Low Groundedness Samples)")
st.write("Reviewing samples where the model might have hallucinated (Score < 4):")

errors = df[df['groundedness'] < 4][['id', 'question', 'groundedness', 'relevance', 'reasoning']]
if not errors.empty:
    st.dataframe(errors, use_container_width=True)
else:
    st.success("No critical hallucinations detected! All samples scored 4 or higher.")

# Download
st.sidebar.markdown("---")
st.sidebar.download_button(
    label="Export Gen Results",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='generation_evaluation.csv',
    mime='text/csv'
)