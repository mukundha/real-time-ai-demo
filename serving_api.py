from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
NUM_MOVIES=9724
# Load the trained model
model = load_model('model/movie_recommendation.h5')
realtime_model = load_model('model/movie_recommendation_realtime.h5')
movies = pd.read_csv('data/movies.csv')
# Create the Flask app
app = Flask(__name__)
CORS(app)

# Define the route for making recommendations
@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    # Get the user ID from the request data
    user_id = request.args.get('user_id')
    realtime = request.args.get('realtime')
    num_recommendations = request.args.get('num_recommendations')
    user_last_watched = request.args.get('user_last_watched')
    user_last_rating = request.args.get('user_last_rating')
    print(user_id,num_recommendations)
    movie_data = []
    # Generate recommendations for the user
    if (realtime == 'true'):
        recommendations = get_user_recommendations_realtime(user_id,user_last_watched,user_last_rating,int(num_recommendations))
    else:
        recommendations = get_user_recommendations(user_id, int(num_recommendations))

    for r in recommendations:        
        if not movies[movies.movieId==r].empty:
            movie = {
                'movieId': int(movies.movieId[movies.movieId==r].iat[0]),
                'poster': movies.poster[movies.movieId==r].iat[0],
                'title': movies.title[movies.movieId==r].iat[0],
                'genres': movies.genres[movies.movieId==r].iat[0]
            }
            movie_data.append(movie)
    print(movie_data)
    # Return the recommendations as a JSON response
    response = {'recommendations': movie_data}
    return jsonify(response)

# Generate movie recommendations for a given user
def get_user_recommendations(user_id=1, num_recommendations=5):
    print(num_recommendations)
    all_movies = np.arange(NUM_MOVIES)
    user = np.full_like(all_movies, user_id)    
    predictions = model.predict([user, all_movies])
    sorted_indices = np.argsort(predictions.flatten())[::-1]
    print(predictions.shape)
    print(sorted_indices.shape)
    top_movies = sorted_indices[:num_recommendations]
    return top_movies

def get_user_recommendations_realtime(user_id=1,user_last_watched=2,user_last_rating=5, num_recommendations=5):
    all_movies = np.arange(NUM_MOVIES)    
    user = np.full_like(all_movies, user_id)        
    predictions = realtime_model.predict([user, all_movies, np.full_like(all_movies, user_last_watched), np.full_like(all_movies, user_last_rating)])
    sorted_indices = np.argsort(predictions.flatten())[::-1]
    top_movies = sorted_indices[:num_recommendations]
    return top_movies

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
