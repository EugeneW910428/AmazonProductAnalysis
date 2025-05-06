
import streamlit as st
from utils.preprocessing import preprocess_text
from utils.model import analyze_sentiment, evaluate_model, train_model
from utils.sql_queries import init_db, insert_reviews, fetch_reviews_by_asin, fetch_top_positive_asins
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Initialize database
init_db()

st.set_page_config(page_title="NextWave Amazon Review Analyzer", layout="wide")

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Upload & Analyze", "Model Evaluation", "Top Products", "About"])

if page == "Upload & Analyze":
    st.title("Amazon Review Analyzer")
    uploaded_file = st.file_uploader("Upload your Amazon reviews (.jsonl)", type="jsonl")

    if uploaded_file:
        data = pd.DataFrame([eval(line) for line in uploaded_file])
        asin = st.text_input("Enter Product ASIN:")

        if asin:
            data = data[data['asin'] == asin]
            data['cleaned_text'] = data['text'].apply(preprocess_text)
            data['sentiment_score'] = data['cleaned_text'].apply(analyze_sentiment)

            # Store to DB
            insert_reviews(data)

            st.subheader(f"Sentiment Analysis for ASIN: {asin}")
            avg_rating = data['rating'].mean()
            sentiment_mean = data['sentiment_score'].mean()

            sentiment_summary = "Positive" if sentiment_mean > 0 else "Negative"
            st.markdown(f"**Average Rating**: {avg_rating:.2f}")
            st.markdown(f"**Overall Sentiment**: {sentiment_summary}")

            wordcloud = WordCloud(width=800, height=400).generate(" ".join(data['cleaned_text']))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

elif page == "Model Evaluation":
    st.title("📊 Model Evaluation")
    data = fetch_reviews_by_asin(None)
    model, report, matrix_fig = train_model(data)
    st.subheader("Classification Report")
    st.text(report)
    st.subheader("Confusion Matrix")
    st.pyplot(matrix_fig)

elif page == "Top Products":
    st.title("Top Products by Positive Sentiment")
    results = fetch_top_positive_asins()
    st.dataframe(results)

elif page == "About":
    st.title("About This App")
    st.write("""
        This tool analyzes Amazon product reviews using NLP and sentiment analysis.
        It demonstrates SQL querying, ML classification, and interactive visualizations.
    """)
