import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

ratings_df = pd.read_csv('data/ratings.csv')
ratings_df['userId'] = ratings_df['userId'].astype('category').cat.codes
ratings_df['movieId'] = ratings_df['movieId'].astype('category').cat.codes
scaler = MinMaxScaler()
ratings_df['rating'] = scaler.fit_transform(ratings_df['rating'].values.reshape(-1, 1))
train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)

num_users = ratings_df['userId'].nunique() + 1
num_movies = ratings_df['movieId'].nunique() + 1

user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))

user_embedding = Embedding(num_users, 50, embeddings_regularizer=l2(1e-5))(user_input)
movie_embedding = Embedding(num_movies, 50, embeddings_regularizer=l2(1e-5))(movie_input)
user_flat = Flatten()(user_embedding)
movie_flat = Flatten()(movie_embedding)
concat = Concatenate()([user_flat, movie_flat])

dense1 = Dense(64, activation='relu', kernel_regularizer=l2(1e-5))(concat)
dense2 = Dense(32, activation='relu', kernel_regularizer=l2(1e-5))(dense1)
output = Dense(1, activation='sigmoid')(dense2)

model = Model(inputs=[user_input, movie_input], outputs=output)
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
model.fit([train_data['userId'].values, train_data['movieId'].values], train_data['rating'].values, epochs=10, batch_size=64)
model.save('movie_recommendation.h5')