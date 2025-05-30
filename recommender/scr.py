import pandas as pd
from surprise import SVD, Dataset, Reader
import joblib

# Load your ratings data
ratings_df = pd.read_csv("data/ratings.csv")

# Prepare the data for Surprise
reader = Reader(rating_scale=(ratings_df['rating'].min(), ratings_df['rating'].max()))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

# Train the SVD model
algo = SVD()
algo.fit(trainset)

# Save the trained model
joblib.dump(algo, "collaborative_model.pkl")
print("Model saved as collaborative_model.pkl")