import streamlit as st
import pandas as pd
from recommender.collaborative import get_top_n_recommendations_actual
from recommender.content_based import get_content_based_recommendations

@st.cache_data
def get_hybrid_recommendations_actual(
    user_id,
    movie_name,
    movies_df,
    ratings_df,
    _collaborative_model,  # already fixed
    _tfidf_matrix,         # <-- add underscore
    _cosine_sim,           # <-- add underscore
    _indices,              # <-- add underscore
    n=10
):
    """
    Generates hybrid recommendations by combining collaborative and content-based approaches.
    """
    st.info(f"Generating hybrid recommendations for User ID {user_id} and movie '{movie_name}'.")

    # Get recommendations
    collab_recs = get_top_n_recommendations_actual(
        user_id, movies_df, ratings_df, _collaborative_model, n=n
    )
    content_recs = get_content_based_recommendations(
        movie_name, movies_df, _tfidf_matrix, _cosine_sim, _indices, n=n
    )

    # Add source identifiers
    collab_recs['source'] = 'Collaborative'
    content_recs['source'] = 'Content-Based'

    # Combine and drop duplicates based on title
    combined_recs = pd.concat([collab_recs, content_recs]).drop_duplicates(subset=['title'])

    # Combine scores
    # Combine scores
    if 'score' in combined_recs.columns and 'final_score' in combined_recs.columns:
        combined_recs['combined_score'] = combined_recs[['score', 'final_score']].max(axis=1)
    elif 'score' in combined_recs.columns:
        combined_recs['combined_score'] = combined_recs['score']
    elif 'final_score' in combined_recs.columns:
        combined_recs['combined_score'] = combined_recs['final_score']
    else:
        combined_recs['combined_score'] = 0.0  # Use 0.0 instead of 1.0 or 10.0

    # Sort and select top N
    final_hybrid_recs = combined_recs.sort_values(by='combined_score', ascending=False).head(n)

    # Ensure required columns
    required_cols = ['title', 'genres', 'overview', 'release_year', 'combined_score']
    for col in required_cols:
        if col not in final_hybrid_recs.columns:
            final_hybrid_recs[col] = "N/A"

    return final_hybrid_recs[required_cols]
