import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Load the ratings_realtime.csv file
ratings_df = pd.read_csv('data/ratings_realtime.csv')

# Convert userId, movieId, and last_watched_movie to categorical variables
ratings_df['userId'] = ratings_df['userId'].astype('category').cat.codes
ratings_df['movieId'] = ratings_df['movieId'].astype('category').cat.codes
ratings_df['last_watched_movie'] = ratings_df['last_watched_movie'].astype('category').cat.codes
ratings_df = ratings_df[ratings_df['last_watched_movie'] >= 0]
# Normalize the ratings
scaler = MinMaxScaler()
ratings_df['rating'] = scaler.fit_transform(ratings_df['rating'].values.reshape(-1, 1))
ratings_df['last_rating'] = scaler.fit_transform(ratings_df['last_rating'].values.reshape(-1, 1))
# Split the data into training and test sets
train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Define the number of unique users and movies
num_users = ratings_df['userId'].nunique() + 1
num_movies = ratings_df['movieId'].nunique() + 1

# Define the input layers
user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))
last_watched_movie_input = Input(shape=(1,))
last_rating_input = Input(shape=(1,))

# Embedding layers for users, movies, last_watched_movie
user_embedding = Embedding(num_users, 50, embeddings_regularizer=l2(1e-5))(user_input)
movie_embedding = Embedding(num_movies, 50, embeddings_regularizer=l2(1e-5))(movie_input)
last_watched_embedding = Embedding(num_movies, 50, embeddings_regularizer=l2(1e-5))(last_watched_movie_input)

# Flatten the embeddings
user_flat = Flatten()(user_embedding)
movie_flat = Flatten()(movie_embedding)
last_watched_flat = Flatten()(last_watched_embedding)

# Concatenate the flattened embeddings and last_rating input
concat = Concatenate()([user_flat, movie_flat, last_watched_flat, last_rating_input])

# Dense layers
dense1 = Dense(64, activation='relu', kernel_regularizer=l2(1e-5))(concat)
dense2 = Dense(32, activation='relu', kernel_regularizer=l2(1e-5))(dense1)
output = Dense(1, activation='sigmoid')(dense2)

# Create the model
model = Model(inputs=[user_input, movie_input, last_watched_movie_input, last_rating_input], outputs=output)
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

# Prepare the training data
train_user = train_data['userId'].values
train_movie = train_data['movieId'].values
train_last_watched_movie = train_data['last_watched_movie'].values
train_last_rating = train_data['last_rating'].values
train_ratings = train_data['rating'].values

# Fit the model
model.fit([train_user, train_movie, train_last_watched_movie, train_last_rating], train_ratings, epochs=100, batch_size=64)

model.save('movie_recommendation_realtime.keras')

