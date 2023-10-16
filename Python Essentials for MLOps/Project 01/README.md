# Movie Recommendation System

This Python script implements a movie recommendation system. It allows users to input the name of a movie, and it provides recommendations based on that input. The recommendations are generated using TF-IDF (Term Frequency-Inverse Document Frequency) to find similar movie titles. Additionally, users can provide a movie ID to find movies that are similar based on user ratings.

The code uses Pandas for data manipulation, Scikit-learn for TF-IDF vectorization, and cosine similarity for recommendation calculations.

## Usage

1. **Requirements:**

   Make sure you have the following dependencies installed:
   - Python 3.x
   - pandas
   - scikit-learn
   - requests
   - numpy

   The movie data is expected to be in CSV files ('data/movies.csv' and 'data/ratings.csv').

2. **Running the Script:**

   To start the movie recommendation system, run the script.

   ```
   python movie_recommendation.py
   ```

3. **Getting Recomendations:**

    Enter the name of a movie to get recommendations based on the movie name. Optionally, you can enter a movie ID to find movies similar to it based on user ratings. Here's an example:


## Linting and Pylint

Linting is the process of checking your code for potential issues, style violations, and errors. To ensure that the code follows good coding practices, we can use Pylint, a Python code linter, by running the following command:

```
pylint movie_recommendation.py
```

he script movie_recommendation.py received a linting score of 10.

## References
[Dataquest - Build a Movie Recommendation System in Python](https://github.com/dataquestio/project-walkthroughs/blob/master/movie_recs/movie_recommendations.ipynb)
