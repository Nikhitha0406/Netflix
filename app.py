import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Apply Custom Styling for Mobile View
page_bg = """
<style>
body {
    background-color: black;
    color: white;
}
.stApp {
    background: rgba(0, 0, 0, 0.9);
    padding: 10px;
}
h1 {
    color: white;
    text-align: center;
    font-size: 40px; /* Reduced size for mobile */
    font-weight: bold;
    margin-bottom: 10px;
    letter-spacing: 2px;
}
h2, label, p, .stMarkdown {
    color: white !important;
    font-size: 14px;
}
.stButton>button {
    background-color: #FF0000;
    color: white;
    font-size: 14px;
    border-radius: 6px;
    width: 100%; /* Full width for better touch interaction */
    padding: 12px;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #cc0000;
}
input, select {
    font-size: 14px;
    padding: 8px;
    width: 100%; /* Full width for mobile */
    background: rgba(255, 255, 255, 0.1);
    border: 2px solid white;
    color: white;
    border-radius: 6px;
}
.netflix-text {
    font-size: 50px; /* Reduced for mobile */
    font-weight: bold;
    text-align: center;
    color: red;
}
.letter {
    display: inline-block;
    opacity: 0;
    animation: fadeInOut 4s infinite;
    font-size: 60px; /* Adjusted size */
}
@keyframes fadeInOut {
    0% { opacity: 0; transform: translateY(-10px); }
    20% { opacity: 1; transform: translateY(0); }
    80% { opacity: 1; }
    100% { opacity: 0; }
}
.movie-title {
    color: #FFD700;
    font-weight: bold;
    font-size: 14px;
}

@media (max-width: 768px) {
    h1 {
        font-size: 35px;
    }
    .letter {
        font-size: 50px;
    }
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# Netflix Movie Recommendation System Title
st.markdown(f"<h1>Netflix Movie Recommender</h1>", unsafe_allow_html=True)

# Netflix Animated Title
netflix_text = "NETFLIX"
animated_netflix = "".join([f'<span class="letter" style="animation-delay:{i*0.5}s;">{char}</span>' for i, char in enumerate(netflix_text)])
st.markdown(f"<h1>🎬 <span class='netflix-text'>{animated_netflix}</span></h1>", unsafe_allow_html=True)

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
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    
    return df.iloc[movie_indices][["title", "description"]]

# Layout for Mobile View
with st.container():
    st.markdown("### 🔍 **Enter or Search a Movie:**")  
    selected_movie = st.selectbox("", [""] + df_movies["title"].tolist(), key="movie_select")

    st.markdown("")

    col1, col2 = st.columns([1, 1])

    with col1:
        search_button = st.button("🍿 Get Recommendations", use_container_width=True)

    with col2:
        clear_button = st.button("❌ Clear", use_container_width=True)

# Recommendation Output
if search_button:
    if selected_movie:
        st.markdown("### 🎥 **Recommended Movies:**")  
        recommendations = recommend_movies(selected_movie)
        
        for index, row in recommendations.iterrows():
            st.markdown(f"<p class='movie-title'>{row['title']}</p><p>{row['description']}</p>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter or select a movie.")

# Clear Button Functionality
if clear_button:
    st.experimental_rerun()
