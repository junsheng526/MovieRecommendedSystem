import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

@st.cache_data
def load_data():
    data = pd.read_csv("MovieDatasetOriginal.csv")
    return data

movies = load_data()

@st.cache_data
def calculate_cosine_sim():
    # Calculate the linear kernel using the TF-IDF matrix
    tfidfv=TfidfVectorizer(analyzer='word', stop_words='english')
    tfidfv_matrix=tfidfv.fit_transform(movies['overview'])
    cosine_sim = linear_kernel(tfidfv_matrix, tfidfv_matrix)
    return cosine_sim

# Calculate and cache the linear kernel
cosine_sim_1 = calculate_cosine_sim()

titles = movies[['original_title']]
indices=pd.Series(data=list(titles.index), index= titles['original_title'] )

def content_based_recommender(title, no_of_recommendations):
    # Get the index of the movie that matches the title
    index = indices[title]
    
    # Get the pairwsie similarity scores of all movies with the selected movie
    sim_scores = list(enumerate(cosine_sim_1[index]))
    
    # Sort the movies based on the similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the top n most similar movies 
    sim_scores = sim_scores[1:no_of_recommendations+1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    recommendations = []
    for (director, year, title) in (movies.iloc[movie_indices][['director', 'release_year', 'original_title']].values):
        recommendations.append((title, year, director))
    
    return recommendations

# Title and Sidebar
st.title("Movies Recommender System")
st.sidebar.header("User Input")

# User Input for User ID and Number of Recommendations
title = st.sidebar.text_input("Enter Title")
no_of_recommendations = st.sidebar.number_input("Number of Recommendations", min_value=1, value=5)

# Create a function to display movie recommendations
def display_recommendations(title, no_of_recommendations):
    st.subheader(f"Recommendations for Title: {title}")

    # Call your content-based recommender function here and get recommendations
    recommendations = content_based_recommender(title, no_of_recommendations)

    if recommendations is None or len(recommendations) == 0:
        st.warning("No recommendations available for this user.")
    else:
        # Display recommendations
        for i, (title, year, director) in enumerate(recommendations, start=1):
            st.write(f"{i}. {title} ({year}) by {director}")

if title == '':
    st.warning(f"Title is null")
else:
    # Display recommendations for the selected user
    display_recommendations(title, no_of_recommendations)