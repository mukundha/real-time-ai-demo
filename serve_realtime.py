import numpy as np
from tensorflow.keras.models import load_model

NUM_MOVIES=9724
model = load_model('recommendation_model.h5')

def get_user_recommendations(user_id,user_last_watched,user_last_rating, num_recommendations):
    all_movies = np.arange(NUM_MOVIES)    
    user = np.full_like(all_movies, user_id)        
    predictions = model.predict([user, all_movies, np.full_like(all_movies, user_last_watched), np.full_like(all_movies, user_last_rating)])
    sorted_indices = np.argsort(predictions.flatten())[::-1]
    top_movies = sorted_indices[:num_recommendations]
    return top_movies

get_user_recommendations(1,1,3,5)