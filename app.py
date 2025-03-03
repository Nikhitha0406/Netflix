import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process  # Replace fuzzywuzzy with rapidfuzz

# Apply Custom Styling
page_bg = """
<style>
body { background-color: black; color: white; }
.stApp { background: rgba(0, 0, 0, 0.9); padding: 20px; }
h1 { color: white; text-align: center; font-size: 50px; font-weight: bold; margin-bottom: 10px; letter-spacing: 3px; }
h2, label, p, .stMarkdown { color: white !important; font-size: 18px; }
.stButton>button { background-color: #FF0000; color: white; font-size: 16px; border-radius: 6px; width: 100%; padding: 8px; transition: 0.3s; }
.stButton>button:hover { background-color: #cc0000; }
input, select { font-size: 10px; padding: 4px; width: 45%; background: rgba(255, 255, 255, 0.1); border: 2px solid white; color: white; border-radius: 6px; }
.movie-title { color: #FFD700; font-weight: bold; font-size: 18px; }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.markdown("<h1>Netflix Movie Recommendation System</h1>", unsafe_allow_html=True)

# Load Dataset
csv_path = "netflix_titles.csv"
if not os.path.exists(csv_path):
    st.error(f"⚠️ File not found: {csv_path}. Ensure it is in the correct directory.")
    st.stop()

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

@st.cache_data
def compute_tfidf_matrix(data):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(data["combined_features"])
    return tfidf_matrix, cosine_similarity(tfidf_matrix)

tfidf_matrix, similarity_matrix = compute_tfidf_matrix(df_movies)

# Function to find closest matching movie
def get_closest_movie(search_title, movie_titles):
    match, score = process.extractOne(search_title, movie_titles)
    return match if score > 60 else None  # Only consider matches with a score > 60

# Function to recommend movies
def recommend_movies(title, df=df_movies, similarity=similarity_matrix):
    title_lower = title.lower()
    movie_titles = df["title"].str.lower().tolist()

    if title_lower not in movie_titles:
        closest_match = get_closest_movie(title, df["title"])
        if closest_match:
            st.info(f"🔍 Did you mean: **{closest_match}**? Showing recommendations for that.")
            title_lower = closest_match.lower()
        else:
            return ["⚠️ Movie not found. Try searching for another one."]

    idx = df[df["title"].str.lower() == title_lower].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]  # Top 5 recommendations
    movie_indices = [i[0] for i in sim_scores]

    return df.iloc[movie_indices][["title", "description"]]

# Search Bar & Button
st.markdown("### 🔍 **Enter or Search a Movie:**")  
selected_movie = st.text_input("Type a movie name:", key="movie_search")

if st.button("🍿 Get Recommendations"):
    if selected_movie:
        st.markdown("### 🎥 **Recommended Movies:**")  
        recommendations = recommend_movies(selected_movie)
        for index, row in recommendations.iterrows():
            st.markdown(f"<p class='movie-title'>{row['title']}</p><p>{row['description']}</p>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a movie name.")
