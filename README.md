### What is Real-time AI?

Applied Machine learning, driven by the most recent, most valuable user and operational data, at scale, to power more accurate decisions and actions in real-time. 
Real-time AI refers to the use of artificial intelligence algorithms and models to make decisions and predictions in real-time with real-time data. 

Real-time AI is used in a wide range of applications, including fraud detection, predictive maintenance, autonomous vehicles, and real-time decision-making in finance, healthcare, and manufacturing. 

### Understanding Real-time in AI 

Let’s take an example of a movie recommendation model to demonstrate and understand the value of Real-time in AI.

We will use [Movielens dataset](https://grouplens.org/datasets/movielens/) to build a recommendation model from scratch, see how it works and evaluate how we could leverage Real-time data to improve the recommendations and user experience.

#### Goal
Given a user, return top 5 movie recommendations

#### Pre-requsities 

```
pip install numpy
pip install pandas
pip install tensorflow
pip install scikit-learn
```

Dataset recommended for educational purpose is already included here. If needed, explore / download [Movielens dataset](https://grouplens.org/datasets/movielens/), 

#### Sample Data

| userId | movieId | rating | timestamp | 
|---- | ------ | ----- | ----| 
| 1 | 1 | 4.0 | 964982703  
| 1 | 3 | 4.0 | 964981247  
| 1 | 6 | 4.0 | 964982224
| 1 | 47| 5.0| 964983815


#### Training

Let's train a recommendation model with this dataset

Feel free to review `train.py` on preprocessing required and the model design.

```
python train.py
```
```
..
Epoch 1/10
1261/1261 [==============================] - 2s 1ms/step - loss: 0.0427
Epoch 2/10
1261/1261 [==============================] - 2s 1ms/step - loss: 0.0379
Epoch 3/10
1261/1261 [==============================] - 2s 1ms/step - loss: 0.0363
Epoch 4/10
1261/1261 [==============================] - 2s 1ms/step - loss: 0.0350
Epoch 5/10
1261/1261 [==============================] - 2s 1ms/step - loss: 0.0338
Epoch 6/10
1261/1261 [==============================] - 2s 1ms/step - loss: 0.0325
Epoch 7/10
1261/1261 [==============================] - 2s 1ms/step - loss: 0.0313
Epoch 8/10
1261/1261 [==============================] - 2s 1ms/step - loss: 0.0303
Epoch 9/10
1261/1261 [==============================] - 2s 1ms/step - loss: 0.0294
Epoch 10/10
1261/1261 [==============================] - 2s 1ms/step - loss: 0.0285
```

After training completes, it will save the trained model to `movie_recommendation.h5`


#### Inference

```
> python serve.py
304/304 [==============================] - 0s 228us/step
[7235 1761  709  277  687]
```
Returns the Top5 movie recommendations for `userId` 1. 

#### Success

Success!! While this is Great, let's talk about how this model could be improved. 

This model uses only the ratings information captured when a user rates a movie for training and inference. What this means is, 

1) The model will produce the same output until its trained on new data.

2) This also misses out on a lot of other actions a user could be doing, for eg, 
- browsing
- checking out trailers
- abandoning a movie midway
- watching a movie but not leaving a rating etc
 these are useful signals that could help with improving the recommendations. 
 
 Many of them are time-bound or values that changes over time, for eg, a feature like `last_watched_movie` will change everytime a user have watched a movie. It will be important to capture the entire journey of such features for better recommendations.  

#### Feature engineering

Now, lets do some feature engineering!

```
Feature engineering or feature extraction or feature discovery is the process of using domain knowledge to extract features (characteristics, properties, attributes) from raw data
```

Notice the timestamp field in our initial dataset. We know when a user had watched a movie. We could use this information to engineer 2 new features

`last_watched_movie` - movie watched by the user before watching `movie_id`

`last_rating`  - movie rating provided to `last_watched_movie`

Also remember, these values change over time. These features are usually referred to as `Real-time` features.

We could populate the new features to our dataset,

|userId|movieId|rating|timestamp|last_watched_movie|last_rating
|--|--|--|--|--|--|
|1|1|4.0|2000-07-30 18:45:03|1023|5.0
|1|3|4.0|2000-07-30 18:20:47|1777|4.0
|1|6|4.0|2000-07-30 18:37:04|2000|4.0
|1|47|5.0|2000-07-30 19:03:35|593|4.0


#### Training

Please review `train_realtime.py` on how we added new features to the model.

```
python train_realtime.py
```

```
..
Epoch 1/10
1253/1253 [==============================] - 3s 2ms/step - loss: 0.0419
Epoch 2/10
1253/1253 [==============================] - 3s 2ms/step - loss: 0.0369
Epoch 3/10
1253/1253 [==============================] - 3s 2ms/step - loss: 0.0351
Epoch 4/10
1253/1253 [==============================] - 3s 2ms/step - loss: 0.0331
Epoch 5/10
1253/1253 [==============================] - 3s 2ms/step - loss: 0.0313
Epoch 6/10
1253/1253 [==============================] - 3s 2ms/step - loss: 0.0297
Epoch 7/10
1253/1253 [==============================] - 3s 2ms/step - loss: 0.0283
Epoch 8/10
1253/1253 [==============================] - 3s 2ms/step - loss: 0.0271
Epoch 9/10
1253/1253 [==============================] - 3s 2ms/step - loss: 0.0261
Epoch 10/10
1253/1253 [==============================] - 3s 2ms/step - loss: 0.0250
```
After training completes, it will save the trained model to `movie_recommendation_realtime.h5`

#### Inference

```
>>> get_user_recommendations(1,1,3,5)
304/304 [==============================] - 0s 258us/step
array([2224,  602,  908,  904, 1700])
>>> get_user_recommendations(1,5,2,5)
304/304 [==============================] - 0s 281us/step
array([ 602, 9600, 2224,  277, 1700])
```

Notice, everytime `Real-time` features changes, in other words, based on user activity, this model is able to generate a more contextual, relevant recommendations. 

This is a simple example. If we are able to capture the user activity in real-time, and use it for inference, we can deliver a contextual, relevant recommendations that will improve the overall user experience. 

Congratulations!! We have built our first `Real-time` AI model!

#### 

Citation
========

To acknowledge use of the dataset in publications, please cite the following paper:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>
