import pandas as pd
import os

# Define paths to your downloaded files
TMDB_MOVIES_PATH = "data/tmdb_5000_movies.csv"
TMDB_CREDITS_PATH = "data/tmdb_5000_credits.csv"
MOVIELENS_RATINGS_PATH = "data/ratings.csv"
MOVIELENS_LINKS_PATH = "data/links.csv"

OUTPUT_MOVIES_PATH = "data/movies.csv"
OUTPUT_RATINGS_PATH = "data/ratings.csv"

print("Starting dataset preparation...")

# --- 1. Load Data ---
try:
    tmdb_movies = pd.read_csv(TMDB_MOVIES_PATH)
    tmdb_credits = pd.read_csv(TMDB_CREDITS_PATH)
    ml_ratings = pd.read_csv(MOVIELENS_RATINGS_PATH)
    ml_links = pd.read_csv(MOVIELENS_LINKS_PATH)
    print("All source datasets loaded.")
except FileNotFoundError as e:
    print(f"Error: One or more source files not found. Please check paths: {e}")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# --- 2. Merge TMDB Movies and Credits ---
tmdb_credits.rename(columns={'movie_id': 'id'}, inplace=True)
tmdb_merged_df = pd.merge(tmdb_movies, tmdb_credits, on='id', how='inner')
print(f"TMDB movies and credits merged. Shape: {tmdb_merged_df.shape}")

# --- 3. Prepare MovieLens Links for Merging ---
ml_links['tmdbId'] = pd.to_numeric(ml_links['tmdbId'], errors='coerce').fillna(0).astype(int)
ml_links_filtered = ml_links[ml_links['tmdbId'].isin(tmdb_merged_df['id'])].copy()

# Add MovieLens movieId to the TMDB merged DataFrame
final_movies_df = pd.merge(tmdb_merged_df, ml_links_filtered[['movieId', 'tmdbId']],
                           left_on='id', right_on='tmdbId', how='left')

# Rename columns for clarity and compatibility
final_movies_df.rename(columns={'id': 'movieId', 'movieId': 'ml_movieId'}, inplace=True)
final_movies_df.drop(columns=['tmdbId'], errors='ignore', inplace=True)

# Ensure movieId is the unique identifier for the movies in this DataFrame
final_movies_df.set_index('movieId', inplace=True)
print(f"Final movies DataFrame prepared. Shape: {final_movies_df.shape}")

# --- 4. Align MovieLens Ratings with TMDB IDs ---
ratings_with_tmdb_id = pd.merge(ml_ratings, ml_links[['movieId', 'tmdbId']], on='movieId', how='inner')
valid_tmdb_ids = final_movies_df.index.unique()
final_ratings_df = ratings_with_tmdb_id[ratings_with_tmdb_id['tmdbId'].isin(valid_tmdb_ids)].copy()

# Drop the original 'movieId' and rename 'tmdbId' to 'movieId' for consistency with app.py
final_ratings_df.drop(columns=['movieId'], inplace=True)
final_ratings_df.rename(columns={'tmdbId': 'movieId'}, inplace=True)

print(f"Final ratings DataFrame prepared. Shape: {final_ratings_df.shape}")

# --- 5. Save the Prepared DataFrames ---
os.makedirs('data', exist_ok=True)
final_movies_df.reset_index(inplace=True)

# Ensure there is a 'title' column for downstream code
if 'title' in final_movies_df.columns:
    pass  # Already correct
elif 'title_x' in final_movies_df.columns:
    final_movies_df.rename(columns={'title_x': 'title'}, inplace=True)
elif 'title_y' in final_movies_df.columns:
    final_movies_df.rename(columns={'title_y': 'title'}, inplace=True)
else:
    raise Exception("No title column found in movies dataframe!")

# Now 'movieId' is the TMDB id, and 'ml_movieId' is the MovieLens id
final_movies_df.to_csv(OUTPUT_MOVIES_PATH, index=False)
final_ratings_df.to_csv(OUTPUT_RATINGS_PATH, index=False)

print(f"Cleaned 'movies.csv' (TMDB-enriched) saved to '{OUTPUT_MOVIES_PATH}'")
print(f"Cleaned 'ratings.csv' (TMDB-aligned) saved to '{OUTPUT_RATINGS_PATH}'")
print("\nDataset preparation complete. You can now run your Streamlit app.")