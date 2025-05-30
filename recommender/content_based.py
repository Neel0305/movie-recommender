import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

@st.cache_data(ttl=86400)
def clean_and_combine_features(movies_df_raw):
    """
    Cleans and combines various text features into a single 'soup' column
    for content-based recommendations. Also prepares rating columns.
    """
    movies_df = movies_df_raw.copy()
    # Ensure 'title' column exists
    if 'movie_title' in movies_df.columns and 'title' not in movies_df.columns:
        movies_df.rename(columns={'movie_title': 'title'}, inplace=True)

    movies_df['genres'] = movies_df['genres'].fillna('').astype(str)
    for col in ['overview', 'keywords', 'cast', 'director', 'vote_average', 'vote_count', 'release_date', 'crew']:
        if col not in movies_df.columns:
            if col in ['overview', 'keywords', 'cast', 'director', 'crew']:
                movies_df[col] = ''
            elif col in ['vote_average', 'vote_count']:
                movies_df[col] = 0.0
            elif col == 'release_date':
                movies_df[col] = None

    movies_df['vote_average'] = pd.to_numeric(movies_df['vote_average'], errors='coerce').fillna(0.0)
    movies_df['vote_count'] = pd.to_numeric(movies_df['vote_count'], errors='coerce').fillna(0.0)
    movies_df['release_year'] = pd.to_datetime(movies_df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)

    def parse_json_string(text):
        try:
            if isinstance(text, str) and text.strip().startswith('['):
                return ' '.join([d['name'] for d in eval(text)])
            return text
        except (SyntaxError, TypeError, ValueError):
            return text

    for feature in ['keywords', 'cast', 'genres']:
        movies_df[feature] = movies_df[feature].apply(parse_json_string)

    def get_director_name(row):
        if isinstance(row['crew'], str) and row['crew'].strip().startswith('['):
            try:
                crew_list = eval(row['crew'])
                directors = [d['name'] for d in crew_list if d['job'] == 'Director']
                return ' '.join(directors)
            except (SyntaxError, TypeError, ValueError):
                pass
        if isinstance(row['director'], str) and row['director'].strip().startswith('['):
            try:
                director_list = eval(row['director'])
                return ' '.join([d['name'] for d in director_list])
            except (SyntaxError, TypeError, ValueError):
                pass
        return row['director']

    movies_df['director'] = movies_df.apply(get_director_name, axis=1)
    movies_df['director'] = movies_df['director'].fillna('')

    movies_df['soup'] = (
        movies_df['overview'] + ' ' +
        movies_df['keywords'] + ' ' +
        movies_df['cast'] + ' ' +
        movies_df['director'] + ' ' +
        movies_df['genres']
    )
    movies_df['soup'] = movies_df['soup'].fillna('')

    return movies_df

@st.cache_resource
def precompute_content_models(df):
    st.info("Precomputing TF-IDF and Cosine Similarity (this happens once).")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    return tfidf, tfidf_matrix, cosine_sim, indices

@st.cache_data
def get_content_based_recommendations(
    movie_title, movies_df, _tfidf_matrix, _cosine_sim, _indices, n=10, rating_weight=0.2
):
    st.info(f"Generating content-based recommendations for '{movie_title}'.")
    idx = _indices.get(movie_title)
    if idx is None:
        return pd.DataFrame()

    sim_scores = list(enumerate(_cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    num_candidates = min(len(sim_scores) - 1, 50)
    candidate_indices = [i[0] for i in sim_scores[1:num_candidates+1]]

    candidate_df = movies_df.iloc[candidate_indices].copy()
    candidate_df['content_similarity'] = [score for idx, score in sim_scores[1:num_candidates+1]]

    if 'vote_average' in movies_df.columns and 'vote_count' in movies_df.columns:
        C = movies_df['vote_average'].mean()
        m = movies_df['vote_count'].quantile(0.90)
        def calculate_weighted_rating_for_rec(x):
            v = x['vote_count']
            R = x['vote_average']
            if v >= m:
                return (v / (v + m) * R) + (m / (v + m) * C)
            else:
                return R
        candidate_df['weighted_rating'] = candidate_df.apply(calculate_weighted_rating_for_rec, axis=1)
        max_weighted_rating = movies_df['weighted_rating'].max() if 'weighted_rating' in movies_df.columns and movies_df['weighted_rating'].max() > 0 else 1.0
        candidate_df['final_score'] = (candidate_df['content_similarity'] * (1 - rating_weight)) + \
                                      (candidate_df['weighted_rating'] / max_weighted_rating * rating_weight)
    else:
        candidate_df['final_score'] = candidate_df['content_similarity']

    required_cols = ['title', 'genres', 'overview', 'release_year', 'final_score']
    for col in required_cols:
        if col not in candidate_df.columns:
            candidate_df[col] = "N/A"

    return candidate_df.sort_values(by='final_score', ascending=False).head(n)