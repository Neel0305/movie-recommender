import streamlit as st
import pandas as pd
import requests
import re
from fuzzywuzzy import process

st.set_page_config(page_title="Movie Recommender", layout="wide", initial_sidebar_state="collapsed")

# Import functions from your recommender module
from recommender.collaborative import load_collaborative_model, get_top_n_recommendations_actual
from recommender.content_based import clean_and_combine_features, precompute_content_models, get_content_based_recommendations
from recommender.hybrid import get_hybrid_recommendations_actual

# --- Configuration ---
import os
TMDB_API_KEY = os.getenv("TMDB_API_KEY")  # Replace with your TMDb API key

# --- Helper Functions (for TMDb API and Fuzzy Matching) ---

def clean_title(title):
    # Remove anything in parentheses (like year)
    return re.sub(r"\s*\(.*\)\s*", "", title).strip()

@st.cache_data(ttl=3600)
def get_tmdb_poster(title):
    """
    Fetches the poster URL for a given movie title from TMDb.
    Tries cleaned and original title for better match.
    """
    search_url = f"https://api.themoviedb.org/3/search/movie"
    for query in [clean_title(title), title]:
        params = {
            "api_key": TMDB_API_KEY,
            "query": query,
        }
        try:
            response = requests.get(search_url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            results = data.get("results")
            if results and results[0].get("poster_path"):
                poster_path = results[0].get("poster_path")
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        except requests.exceptions.RequestException:
            continue
    return None

@st.cache_data
def get_movie_title_suggestions(input_title, all_titles, limit=5):
    """
    Finds the closest matching movie titles using fuzzy matching.
    """
    matches = process.extract(input_title, all_titles, limit=limit)
    return [title for title, score in matches if score >= 80]

# --- Data Loading (Cached) ---

@st.cache_data(ttl=86400)
def load_raw_data():
    """Loads movie and ratings dataframes, with error handling."""
    try:
        movies_df = pd.read_csv("data/movies.csv")
        ratings_df = pd.read_csv("data/ratings.csv")
        return movies_df, ratings_df
    except FileNotFoundError:
        st.error("Error: 'movies.csv' or 'ratings.csv' not found in the 'data/' directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        st.stop()

# Load and process data globally for the app
movies_df_raw, ratings_df = load_raw_data()
processed_movies_df = clean_and_combine_features(movies_df_raw)


# Check for 'title' column
if 'title' not in processed_movies_df.columns:
    st.error("Error: 'title' column missing from processed_movies_df!")
    st.stop()

# Precompute content-based models globally for the app
tfidf_model, tfidf_matrix_global, cosine_sim_global, indices_global = precompute_content_models(processed_movies_df)

# Load collaborative model globally for the app
collaborative_model = load_collaborative_model()

# --- Streamlit App Layout ---


st.title("🎬 Movie Recommendation System")
st.markdown("Get personalized movie recommendations using **Collaborative Filtering**, **Content-Based (NLP)**, or a **Hybrid** approach.")

# Global settings
st.sidebar.header("⚙️ Settings")
top_n_recommendations = st.sidebar.slider("Number of Recommendations:", min_value=5, max_value=20, value=10, step=1)
st.markdown("---")

# --- Display Recommendations Function ---
def display_recommendations(recs_df):
    if recs_df.empty:
        st.warning("No recommendations found. Please try different inputs.")
        return

    st.write(f"Displaying top {len(recs_df)} recommendations:")
    st.download_button("Download as CSV", recs_df.to_csv(index=False), file_name="recommendations.csv")
    for i, row in recs_df.iterrows():
        poster_url = get_tmdb_poster(row['title'])
        col1, col2 = st.columns([1, 4])
        with col1:
            if poster_url:
                st.image(poster_url, width=100, caption=f"Poster for {row['title']}")
            else:
                st.info("No poster available.")
        with col2:
            st.markdown(f"**🎥 {row['title']}** ({row.get('release_year', 'N/A')})")
            st.markdown(f"*{row['genres']}*")
            if 'overview' in row and row['overview'] and row['overview'] != "N/A" and row['overview'] != "":
                with st.expander("See Overview"):
                    st.write(row['overview'])
            # Display score if available
            if 'score' in row:
                st.write(f"Relevance Score: **{row['score']:.3f}**")
            elif 'final_score' in row:
                st.write(f"Relevance Score: **{row['final_score']:.3f}**")
            elif 'combined_score' in row:
                st.write(f"Relevance Score: **{row['combined_score']:.3f}**")

# --- Collaborative Filtering Section ---
if option := st.radio("📌 Choose Recommendation Type", ["Collaborative", "Content-Based", "Hybrid"], horizontal=True):
    st.markdown("---")

    if option == "Collaborative":
        st.subheader("👤 Collaborative Filtering")
        user_id = st.number_input(
            "Enter your User ID:",
            min_value=int(ratings_df['userId'].min()),
            max_value=int(ratings_df['userId'].max()),
            value=1,
            help="Enter a User ID from your ratings dataset (e.g., 1 to 610 for MovieLens Small)"
        )

        if st.button("Recommend Movies"):
            with st.spinner("Generating recommendations..."):
                recs = get_top_n_recommendations_actual(user_id, processed_movies_df, ratings_df, collaborative_model, n=top_n_recommendations)
                if not recs.empty:
                    st.success(f"Here are your movie recommendations for User ID {user_id}:")
                    display_recommendations(recs)
                else:
                    st.warning(f"Could not find recommendations for User ID {user_id}. This user might not have enough ratings or no unrated movies.")

    # --- Content-Based Filtering Section ---
    elif option == "Content-Based":
        st.subheader("🎯 Content-Based Filtering")
        movie_input = st.text_input("Enter a movie title you like:", "Toy Story")
        all_movie_titles = processed_movies_df['title'].unique().tolist()
        movie_to_recommend = None

        if movie_input:
            suggestions = get_movie_title_suggestions(movie_input, all_movie_titles)
            if suggestions:
                st.markdown("Did you mean one of these? (Select the exact title for best results)")
                selected_movie_from_suggestions = st.radio("Suggestions:", suggestions, key="content_suggestions")
                movie_to_recommend = selected_movie_from_suggestions
            else:
                st.info("No close matches found. Please refine your search or check spelling.")
                movie_to_recommend = None
        else:
            st.info("Please enter a movie title.")

        if st.button("Find Similar Movies") and movie_to_recommend:
            with st.spinner(f"Searching similar movies to '{movie_to_recommend}'..."):
                recs = get_content_based_recommendations(
                    movie_to_recommend,
                    processed_movies_df,
                    tfidf_matrix_global,
                    cosine_sim_global,
                    indices_global,
                    n=top_n_recommendations
                )
                if not recs.empty:
                    st.success(f"You might also like movies similar to '{movie_to_recommend}':")
                    display_recommendations(recs)
                else:
                    st.warning(f"Could not find similar movies for '{movie_to_recommend}'.")
        elif st.button("Find Similar Movies"):
            st.warning("Please enter a movie title to find similar movies.")

    # --- Hybrid Recommendations Section ---
    elif option == "Hybrid":
        st.subheader("🔀 Hybrid Recommendations")
        user_id_hybrid = st.number_input(
            "Enter your User ID:",
            min_value=int(ratings_df['userId'].min()),
            max_value=int(ratings_df['userId'].max()),
            value=1,
            key="hybrid_user_id",
            help="Enter a User ID from your ratings dataset"
        )
        movie_input_hybrid = st.text_input("Enter a movie title you like (for content component):", "Jumanji", key="hybrid_movie_input")
        all_movie_titles_hybrid = processed_movies_df['title'].unique().tolist()
        movie_to_recommend_hybrid = None

        if movie_input_hybrid:
            suggestions_hybrid = get_movie_title_suggestions(movie_input_hybrid, all_movie_titles_hybrid)
            if suggestions_hybrid:
                st.markdown("Did you mean one of these? (Select the exact title for best results)")
                selected_movie_from_suggestions_hybrid = st.radio("Suggestions:", suggestions_hybrid, key="hybrid_suggestions")
                movie_to_recommend_hybrid = selected_movie_from_suggestions_hybrid
            else:
                st.info("No close matches found. Please refine your search or check spelling.")
                movie_to_recommend_hybrid = None
        else:
            st.info("Please enter a movie title for the content component.")

        # Only show one button at a time
        if movie_to_recommend_hybrid:
            if st.button("Get Hybrid Recommendations", key="hybrid_recommend_btn"):
                with st.spinner("Combining both systems..."):
                    recs = get_hybrid_recommendations_actual(
                        user_id_hybrid,
                        movie_to_recommend_hybrid,
                        processed_movies_df,
                        ratings_df,
                        collaborative_model,
                        tfidf_matrix_global,
                        cosine_sim_global,
                        indices_global,
                        n=top_n_recommendations
                    )
                    if not recs.empty:
                        st.success("Blended recommendations just for you:")
                        display_recommendations(recs)
                    else:
                        st.warning(f"Could not generate hybrid recommendations for User ID {user_id_hybrid} and movie '{movie_to_recommend_hybrid}'.")
        else:
            if st.button("Get Hybrid Recommendations", key="hybrid_recommend_btn_warn"):
                st.warning("Please enter a movie title for the content component.")

# --- Footer ---
st.markdown("---")
st.markdown("🚀 *Built with Streamlit, Scikit-learn, Scikit-Surprise, and TMDb API.*")

# Custom background and style
st.markdown(
    """
    <style>
    .stApp {
        background: url('https://images.unsplash.com/photo-1464983953574-0892a716854b?auto=format&fit=crop&w=1200&q=80');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: #fff;
    }
    .block-container {
        background: rgba(30,30,30,0.85);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31,38,135,0.37);
        backdrop-filter: blur(4px);
        margin-top: 2rem;
    }
    h1, h2, h3, h4 {
        color: #ffb347;
        font-family: 'Segoe UI', sans-serif;
        text-shadow: 2px 2px 8px #232526;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(135deg, #232526 0%, #414345 100%);
        color: #fff;
    }
    button[kind="primary"] {
        background: linear-gradient(90deg, #ffb347 0%, #ffcc33 100%);
        color: #232526;
        border-radius: 8px;
        font-weight: bold;
    }
    /* Poster style for images */
    img {
        border-radius: 12px !important;
        box-shadow: 0 4px 24px 0 rgba(0,0,0,0.4), 0 1.5px 4px 0 rgba(255,180,71,0.15);
        border: 3px solid #ffb34733;
        margin-bottom: 0.5rem;
        transition: transform 0.2s;
    }
    img:hover {
        transform: scale(1.04);
        border-color: #ffb347;
    }
    </style>
    """,
    unsafe_allow_html=True
)