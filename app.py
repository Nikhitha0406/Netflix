import streamlit as st
import pandas as pd
import zipfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define paths
zip_path = "netflix_titles.csv.zip"
extract_folder = "extracted_files"
csv_path = os.path.join(extract_folder, "netflix_titles.csv")

# Extract ZIP file if not already extracted
if not os.path.exists(extract_folder):
    os.makedirs(extract_folder)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)

# Load dataset with caching
@st.cache_data
def load_data():
    df = pd.read_csv(csv_path)
    df.fillna({"description": "", "listed_in": "", "director": "", "cast": ""}, inplace=True)
    df_movies = df[df["type"] == "Movie"].copy()
    df_movies["combined_features"] = (
        df_movies["listed_in"] + " " +
        df_movies["director"] + " " +
        df_movies["cast"] + " " +
        df_movies["description"]
    )
    return df_movies

df_movies = load_data()

# Compute TF-IDF with caching
@st.cache_data
def compute_tfidf_matrix(data):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(data["combined_features"])
    return tfidf_matrix, cosine_similarity(tfidf_matrix)

tfidf_matrix, similarity_matrix = compute_tfidf_matrix(df_movies)

# Function to recommend movies
def recommend_movies(title, df=df_movies, similarity=similarity_matrix):
    indices = df[df["title"].str.lower() == title.lower()].index
    if len(indices) == 0:
        return ["Movie not found. Please try another title."]
    idx = indices[0]

    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]  # Top 5 recommendations
    movie_indices = [i[0] for i in sim_scores]
    
    return df.iloc[movie_indices][["title", "description"]]

# Streamlit UI
st.title("🎬 Netflix Movie Recommendation System")

# Option 1: Text Input
movie_name = st.text_input("Enter a movie title:")

# Option 2: Dropdown for better UX
movie_list = df_movies["title"].tolist()
selected_movie = st.selectbox("Or select a movie:", [""] + movie_list)

# Button to get recommendations
if st.button("Get Recommendations"):
    query_movie = selected_movie if selected_movie else movie_name
    if query_movie:
        recommendations = recommend_movies(query_movie)
        st.subheader("Recommended Movies:")
        for index, row in recommendations.iterrows():
            st.write(f"**{row['title']}** - {row['description']}")
    else:
        st.warning("Please enter or select a movie.")
