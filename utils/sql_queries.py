import sqlite3
import pandas as pd

DB_NAME = 'reviews.db'

# Create table if not exists
CREATE_SQL = """
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asin TEXT,
    rating INTEGER,
    text TEXT,
    cleaned_text TEXT,
    sentiment_score REAL
);
"""

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute(CREATE_SQL)
        conn.commit()

def insert_reviews(df):
    with sqlite3.connect(DB_NAME) as conn:
        df[['asin', 'rating', 'text', 'cleaned_text', 'sentiment_score']].to_sql('reviews', conn, if_exists='append', index=False)

def fetch_reviews_by_asin(asin=None):
    with sqlite3.connect(DB_NAME) as conn:
        if asin:
            query = "SELECT * FROM reviews WHERE asin = ?"
            return pd.read_sql(query, conn, params=(asin,))
        else:
            return pd.read_sql("SELECT * FROM reviews", conn)

def fetch_top_positive_asins(limit=5):
    with sqlite3.connect(DB_NAME) as conn:
        query = """
        SELECT asin, COUNT(*) as positive_count
        FROM reviews
        WHERE sentiment_score > 0.2
        GROUP BY asin
        ORDER BY positive_count DESC
        LIMIT ?
        """
        return pd.read_sql(query, conn, params=(limit,))
