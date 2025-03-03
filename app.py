import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Apply custom CSS for an eye-catching design
st.markdown("""
    <style>
        body {
            background-color: #0d0d0d;
            color: white;
        }
        .stApp {
            background-color: #111;
        }
        .title {
            font-size: 50px;
            font-weight: bold;
            text-align: center;
            color: #E50914;
            text-shadow: 3px 3px 5px #ff0000;
        }
        .subheader {
            font-size: 30px;
            font-weight: bold;
            text-align: center;
            color: #FFD700;
            text-shadow: 2px 2px 5px #FFA500;
        }
        .movie-card {
            background: linear-gradient(135deg, #ff6600, #ff0000);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
            box-shadow: 3px 3px 15px rgba(255, 0, 0, 0.5);
            transition: transform 0.3s ease-in-out;
        }
        .movie-card:hover {
            transform: scale(1.05);
        }
        .movie-title {
            font-size: 22px;
            font-weight: bold;
            color: #FFFFFF;
        }
        .movie-description {
            font-size: 18px;
            color: #F5F5F5;
        }
        .stTextInput > div > div > input {
            border: 3px solid #FFD700;
            background-color: #222;
            color: white;
            font-size: 18px;
        }
        .stSelectbox > div > div {
            border: 3px solid #FFD700;
            background-color: #222;
            color: white;
            font-size: 18px;
        }
        .stButton > button {
            background: linear-gradient(135deg, #E50914, #ff6600);
            color: white;
            border-radius: 12px;
            padding: 15px 25px;
            font-size: 20px;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #ff0000, #ff8800);
            transform: scale(1.1);
        }
    </style>
""", unsafe_allow_html=True)

# Define CSV file path
csv_path = "netflix_titles.csv"

# Check if file exists
if not os.path.exists(csv_path):
    st.error(f"⚠️ File not found: {csv_path}. Ensure it is in the correct directory.")
    st.stop()

# Load dataset with caching
@st.cache_data
def load_data():
    df = pd.read_csv(csv_path, encoding="utf-8")
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
st.markdown('<p class="title">🎬 Netflix Movie Recommendation System</p>', unsafe_allow_html=True)

# Movie input options
st.markdown('<p class="subheader">🔥 Find Movies Similar to Your Favorites 🔥</p>', unsafe_allow_html=True)
movie_name = st.text_input("🔍 Enter a movie title:")
movie_list = df_movies["title"].tolist()
selected_movie = st.selectbox("🎞️ Or select a movie:", [""] + movie_list)

# Button to get recommendations
if st.button("🎥 Get Recommendations"):
    query_movie = selected_movie if selected_movie else movie_name
    if query_movie:
        recommendations = recommend_movies(query_movie)
        st.markdown('<p class="subheader">✨ Recommended Movies ✨</p>', unsafe_allow_html=True)
        
        # Display movies in styled cards
        for index, row in recommendations.iterrows():
            st.markdown(f"""
                <div class="movie-card">
                    <p class="movie-title">🎬 {row['title']}</p>
                    <p class="movie-description">{row['description']}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter or select a movie.")
