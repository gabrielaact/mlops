# import libraries
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import logging

# configure the logging settings
logging.basicConfig(filename='movie_recommendation.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# read data
def read_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        print("File not found!")
        return None
    except Exception as e:
        logging.error("An error occurred while reading the file: %s", e)
        print("An error occurred while reading the file:", e)
        return None
    
# clean the movie title
def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title

# create TF-IDF matrix
def create_tfidf_matrix(movies_df):
    movies_df["clean_title"] = movies_df["title"].apply(clean_title)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(movies_df["clean_title"])

    return tfidf, vectorizer

# obtain the five most similar titles to our search term
def get_recommendations(movies_df, title, num_recommendations=5):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argsort(similarity)[::-1][:num_recommendations]
    results = movies_df.iloc[indices]
    
    return results

# find 10 similar movies by user ratings
def find_similar_movies(ratings_df, movie_id):
    similar_users = ratings_df[(ratings_df["movieId"] == movie_id) & (ratings_df["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings_df[(ratings_df["userId"].isin(similar_users)) & (ratings_df["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings_df[(ratings_df["movieId"].isin(similar_user_recs.index)) & (ratings_df["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(movies_df, left_index=True, right_on="movieId")[["score", "title", "genres"]]

# user interaction and display of recommendations
def user_interaction():
    while True:
        user_input = input("Enter the name of a movie (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        
        recommendations = get_recommendations(movies_df, user_input)
        
        if not recommendations.empty:
            print("\nMovie recommendations based on '{}':".format(user_input))
            print(recommendations[['title', 'genres']])
        else:
            print("No matching movies found.\n")
        
        logging.info("User input: %s", user_input)  
        
        movie_id_input = input("Enter the movie ID to find similar movies (or 'skip' to continue): ")
        if movie_id_input.lower() == 'skip':
            continue
        
        try:
            movie_id = int(movie_id_input)
            similar_movies = find_similar_movies(ratings_df, movie_id)
            movie_title = movies_df[movies_df["movieId"] == movie_id]["title"].values[0]
            print("\nMovies similar to '{}':".format(movie_title))
            print(similar_movies[["title", "genres"]])
            print("\n")
            logging.info("Movie ID input: %s", movie_id_input)  
        except ValueError:
            print("Invalid movie ID. Please enter a valid movie ID or 'skip'.\n")
            logging.error("Invalid movie ID input: %s", movie_id_input)  

if __name__ == "__main__":
    movies_df = read_data("data/movies.csv")
    ratings_df = read_data("data/ratings.csv")
    [tfidf, vectorizer] = create_tfidf_matrix(movies_df)
    user_interaction()