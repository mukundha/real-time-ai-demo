import numpy as np
from tensorflow.keras.models import load_model

NUM_MOVIES=9724
model = load_model('movie_recommendation.h5')

def get_user_recommendations(user_id, num_recommendations):
    all_movies = np.arange(NUM_MOVIES)    
    user = np.full_like(all_movies, user_id)        
    predictions = model.predict([user, all_movies])
    sorted_indices = np.argsort(predictions.flatten())[::-1]
    top_movies = sorted_indices[:num_recommendations]
    return top_movies

get_user_recommendations(1,5)