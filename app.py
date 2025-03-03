import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Apply Custom Background & Styling
page_bg = """
<style>
body {
    background-color: black;
    color: white;
}
.stApp {
    background: rgba(0, 0, 0, 0.9);
    padding: 50px;
}
h1 {
    color: #FF0000;
    text-align: center;
    font-size: 70px;
    margin-bottom: 40px;
}
h2, h3, label {
    color: white !important;
    font-size: 24px;
    text-align: center;
    margin-bottom: 20px;
}
.stButton>button {
    background-color: #FF0000;
    color: white;
    font-size: 22px;
    border-radius: 10px;
    width: 100%;
    padding: 15px;
    transition: 0.3s;
    margin-top: 20px;
}
.stButton>button:hover {
    background-color: #cc0000;
}
input, select {
    font-size: 20px;
    padding: 15px;
    width: 100%;
    background: rgba(255, 255, 255, 0.1);
    border: 2px solid #FF0000;
    color: white;
    border-radius: 8px;
    text-align: center;
}
.letter {
    font-size: 80px;
    color: #FF0000;
    font-weight: bold;
    display: inline-block;
    animation: swing 1s ease-in-out infinite alternate;
}
@keyframes swing {
    0% { transform: rotate(-10deg); }
    100% { transform: rotate(10deg); }
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Netflix Swinging Animation Title
netflix_text = "NETFLIX"
animated_netflix = "".join([f'<span class="letter" style="animation-delay:{i*0.2}s;">{char}</span>' for i, char in enumerate(netflix_text)])

st.markdown(f"<h1>🎬 Netflix Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown(f"<h1>{animated_netflix}</h1>", unsafe_allow_html=True)

# Add spacing
st.markdown("<br><br>", unsafe_allow_html=True)

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

# **Unified Input Bar**: Type or Select
st.markdown("### 🔍 **Enter or Search a Movie:**")  
movie_list = df_movies["title"].tolist()
selected_movie = st.selectbox("", [""] + movie_list)

# Add spacing
st.markdown("<br>", unsafe_allow_html=True)

# Button to get recommendations
if st.button("🍿 Get Recommendations"):
    st.markdown("<br>", unsafe_allow_html=True)
    if selected_movie:
        st.markdown("### 🎥 **Recommended Movies:**")  
        recommendations = recommend_movies(selected_movie)
        for index, row in recommendations.iterrows():
            st.write(f"**{row['title']}** - {row['description']}")
    else:
        st.warning("⚠️ Please enter or select a movie.")
