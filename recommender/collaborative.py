import streamlit as st
import pandas as pd
import joblib  # For loading a trained model, if needed

@st.cache_resource
def load_collaborative_model():
    """
    Loads the collaborative filtering model.
    """
    st.info("Loading collaborative filtering model...")
    model = joblib.load("collaborative_model.pkl")
    return model

@st.cache_data
def get_top_n_recommendations_actual(user_id, movies_df, ratings_df, _model, n=10):
    st.info(f"Generating collaborative recommendations for User ID {user_id}.")

    user_rated_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    unrated_movies = movies_df[~movies_df['movieId'].isin(user_rated_movie_ids)].copy()

    if unrated_movies.empty:
        return pd.DataFrame(columns=['title', 'genres', 'overview', 'release_year', 'score'])

    def predict_score(row):
        try:
            pred = _model.predict(user_id, row['movieId'])
            return pred.est
        except Exception:
            return 0.0

    unrated_movies['score'] = unrated_movies.apply(predict_score, axis=1)
    recs = unrated_movies.sort_values(by='score', ascending=False).head(n)

    required_cols = ['title', 'genres', 'overview', 'release_year', 'score']
    for col in required_cols:
        if col not in recs.columns:
            recs[col] = "N/A"

    return recs[required_cols]