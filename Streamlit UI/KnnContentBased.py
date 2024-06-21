import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("MovieDatasetOriginal.csv")
    return data

movies = load_data()

# Create TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['overview'])

# Fit Nearest Neighbors model
nn_model = NearestNeighbors(n_neighbors=10, metric='cosine')  # Adjust n_neighbors as needed
nn_model.fit(tfidf_matrix)

# Sidebar and title
st.title("Movies Recommender System")
st.sidebar.header("User Input")

# User Input for Movie Title and Number of Recommendations
title = st.sidebar.text_input("Enter Movie Title")
no_of_recommendations = st.sidebar.number_input("Number of Recommendations", min_value=1, value=5)

# Function to get recommendations using KNN
def get_recommendations(title, no_of_recommendations):
    # Transform input title to TF-IDF vector
    title_vector = tfidf_vectorizer.transform([title])

    # Find nearest neighbors
    distances, indices = nn_model.kneighbors(title_vector)

    # Get recommendations based on nearest neighbors
    recommendations = []
    for idx in indices.flatten()[1:no_of_recommendations+1]:  # Exclude the input movie itself
        movie_title = movies.iloc[idx]['original_title']
        release_year = movies.iloc[idx]['release_year']
        director = movies.iloc[idx]['director']
        recommendations.append((movie_title, release_year, director))

    return recommendations

# Display recommendations
if title.strip() == '':
    st.warning("Please enter a movie title.")
else:
    st.subheader(f"Recommendations for Movie: {title}")
    recommendations = get_recommendations(title, no_of_recommendations)

    if not recommendations:
        st.warning("No recommendations found.")
    else:
        for i, (movie_title, release_year, director) in enumerate(recommendations, start=1):
            st.write(f"{i}. {movie_title} ({release_year}) by {director}")
