import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Apply Custom Background & Styling
page_bg = """
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1574375927938-d5a98e8ffe85?fm=jpg&q=60&w=3000");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: white;
}
.stApp {
    background: rgba(0, 0, 0, 0.85);  /* Dark overlay for better readability */
    padding: 20px;
}
h1, h2, h3 {
    color: #FF0000;  /* Netflix Red */
    text-align: center;
}
.stButton>button {
    background-color: #FF0000;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    width: 100%;
}
input, select {
    font-size: 18px;
    padding: 10px;
    width: 100%;
    background: rgba(255, 255, 255, 0.1);
    border: 2px solid #FF0000;
    color: white;
    border-radius: 5px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Define CSV file path
csv_path = "netflix_titles.csv"

# Check if file exists
if not os.path.exists(csv_path):
    st.error(f"⚠️ File not found: {csv_path}. Ensure it is in the correct directory.")
    st.stop()

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
        return ["⚠️ Movie not found. Please try another title."]
    idx = indices[0]

    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]  # Top 5 recommendations
    movie_indices = [i[0] for i in sim_scores]
    
    return df.iloc[movie_indices][["title", "description"]]

# Streamlit UI
st.title("🎬 **Netflix Movie Recommendation System**")

# **Unified Input Bar**: Type or Select
movie_list = df_movies["title"].tolist()
selected_movie = st.selectbox("🔍 Search or Select a Movie:", [""] + movie_list)

# Button to get recommendations
if st.button("🍿 Get Recommendations"):
    if selected_movie:
        recommendations = recommend_movies(selected_movie)
        st.subheader("🎥 Recommended Movies:")
        for index, row in recommendations.iterrows():
            st.write(f"**{row['title']}** - {row['description']}")
    else:
        st.warning("⚠️ Please enter or select a movie.")
