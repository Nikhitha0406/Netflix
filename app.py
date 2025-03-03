import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data  # Cache the data loading function
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df.fillna("", inplace=True)  # Handle missing values
    df_movies = df[df["type"] == "Movie"].copy()
    df_movies["combined_features"] = (
        df_movies["listed_in"] + " " +
        df_movies["director"] + " " +
        df_movies["cast"] + " " +
        df_movies["description"]
    )
    return df_movies

df_movies = load_data()

# Compute TF-IDF
@st.cache_data  # Cache the vectorizer to improve performance
def compute_tfidf_matrix(data):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(data["combined_features"])
    return tfidf_matrix, cosine_similarity(tfidf_matrix)

tfidf_matrix, similarity_matrix = compute_tfidf_matrix(df_movies)

df_movies.reset_index(drop=True, inplace=True)

# Function to recommend movies
def recommend_movies(title):
    idx = df_movies[df_movies["title"].str.lower() == title.lower()].index
    if len(idx) == 0:
        return ["Movie not found."]
    idx = idx[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]  # Exclude itself
    
    # Retrieve movie titles
    movie_indices = [i[0] for i in sim_scores]
    return df_movies.iloc[movie_indices]["title"].tolist()

# Streamlit UI
st.title("Netflix Movie Recommendation System")

# Provide dropdown for better UX
movie_list = df_movies["title"].tolist()
movie_name = st.selectbox("Select a movie:", [""] + movie_list)

if st.button("Get Recommendations") and movie_name:
    recommendations = recommend_movies(movie_name)
    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write(f"- {movie}")


