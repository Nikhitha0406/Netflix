import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Apply Custom Styling
page_bg = """
<style>
body {
    background-color: black;
    color: white;
}
.stApp {
    background: rgba(0, 0, 0, 0.9);
    padding: 20px;
}
h1 {
    color: white;
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    margin-bottom: 10px;
    letter-spacing: 3px;
}
h2, label, p, .stTextInput, .stSelectbox, .stMarkdown {
    color: white !important;
    font-size: 18px;
}
.stButton>button {
    background-color: #FF0000;
    color: white;
    font-size: 18px;
    border-radius: 6px;
    width: 100%;
    padding: 10px;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #cc0000;
}
input, select {
    font-size: 16px;
    padding: 10px;
    width: 100%;
    background: rgba(255, 255, 255, 0.1);
    border: 2px solid white;
    color: white;
    border-radius: 6px;
}
.netflix-text {
    font-size: 50px;
    font-weight: bold;
    text-align: center;
    color: red;
}
.letter {
    display: inline-block;
    opacity: 0;
    animation: fadeIn 1s forwards;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}
.movie-title {
    color: #FFD700; /* Gold color for movie titles */
    font-weight: bold;
    font-size: 20px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Netflix Animated Title (One-by-One Letter Reveal in Red)
netflix_text = "NETFLIX"
animated_netflix = "".join([f'<span class="letter" style="animation-delay:{i*0.5}s;">{char}</span>' for i, char in enumerate(netflix_text)])

st.markdown(f"<h1>🎬 <span class='netflix-text'>{animated_netflix}</span></h1>", unsafe_allow_html=True)
st.markdown(f"<h1>Netflix Movie Recommendation System</h1>", unsafe_allow_html=True)

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

# **Search/Select Bar & Button Below Each Other**
st.markdown("### 🔍 **Enter or Search a Movie:**")  
selected_movie = st.selectbox("", [""] + df_movies["title"].tolist())

if st.button("🍿 Get Recommendations"):
    if selected_movie:
        st.markdown("### 🎥 **Recommended Movies:**")  
        recommendations = recommend_movies(selected_movie)
        for index, row in recommendations.iterrows():
            st.markdown(f"<p class='movie-title'>{row['title']}</p><p>{row['description']}</p>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter or select a movie.")
