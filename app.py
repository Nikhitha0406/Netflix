import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Apply custom CSS for better styling
st.markdown("""
    <style>
        /* Background & Text */
        body {
            background-color: #141414;
            color: white;
        }
        .stApp {
            background: linear-gradient(135deg, #e50914, #b81d24, #ff6f61);
            padding: 20px;
            border-radius: 15px;
        }
        /* Title */
        .title {
            font-size: 50px;
            font-weight: bold;
            text-align: center;
            color: #FFD700;
            text-shadow: 3px 3px 5px rgba(255, 0, 0, 0.8);
        }
        /* Subheader */
        .subheader {
            font-size: 26px;
            font-weight: bold;
            text-align: center;
            color: #ffffff;
            margin-bottom: 20px;
        }
        /* Movie Cards */
        .movie-card {
            background: linear-gradient(45deg, #202020, #303030);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 15px;
            box-shadow: 4px 4px 10px rgba(255, 0, 0, 0.5);
        }
        .movie-title {
            font-size: 22px;
            font-weight: bold;
            color: #ffcc00;
            text-shadow: 2px 2px 5px rgba(255, 255, 0, 0.8);
        }
        .movie-description {
            font-size: 16px;
            color: #f5f5f5;
        }
        /* Styled Input Fields */
        .stTextInput > div > div > input {
            border: 2px solid #FFD700;
            background-color: #303030;
            color: white;
            font-size: 18px;
            padding: 10px;
        }
        .stSelectbox > div > div {
            border: 2px solid #FFD700;
            background-color: #303030;
            color: white;
            font-size: 18px;
        }
        /* Buttons */
        .stButton > button {
            background: linear-gradient(90deg, #ff4500, #ff0000);
            color: white;
            font-size: 20px;
            font-weight: bold;
            border-radius: 15px;
            padding: 15px 25px;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #ff0000, #ff4500);
            box-shadow: 2px 2px 10px rgba(255, 0, 0, 0.8);
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
    df = pd.read_csv(csv_path, encoding="utf-8")  # Ensure UTF-8 encoding
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
st.markdown('<p class="title">🍿 Netflix Movie Recommendation System 🎬</p>', unsafe_allow_html=True)

# Movie input options
st.markdown('<p class="subheader">🔍 Find Movies Similar to Your Favorites</p>', unsafe_allow_html=True)
movie_name = st.text_input("Enter a movie title:")
movie_list = df_movies["title"].tolist()
selected_movie = st.selectbox("🎞️ Or select a movie:", [""] + movie_list)

# Button to get recommendations
if st.button("🔥 Get Movie Recommendations 🎥"):
    query_movie = selected_movie if selected_movie else movie_name
    if query_movie:
        recommendations = recommend_movies(query_movie)
        st.markdown('<p class="subheader">🎬 Recommended Movies:</p>', unsafe_allow_html=True)
        
        # Display movies in styled cards
        for index, row in recommendations.iterrows():
            st.markdown(f"""
                <div class="movie-card">
                    <p class="movie-title">🎥 {row['title']}</p>
                    <p class="movie-description">{row['description']}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter or select a movie.")
