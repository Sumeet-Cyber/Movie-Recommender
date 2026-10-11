# Movie Recommender

## Overview

This project is a movie recommender built on the MovieLens 100K dataset. It learns from users' past ratings to suggest films they haven't seen yet. The model is a hybrid: it combines collaborative filtering, which learns taste patterns from how users rate movies, with a content-based score built from each movie's genres and release decade.

The project includes a Jupyter notebook that covers data exploration, model training, tuning, and evaluation, and a Streamlit app that lets you pick a user, see their recommendations, and find movies similar to a title you choose.

## Dataset

The project uses [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/), published by GroupLens Research at the University of Minnesota. It contains:

- 100,000 ratings on a 1 to 5 scale
- 943 users, each with at least 20 ratings
- 1,682 movies, each with one or more genre labels and a release date
- Basic user demographics (age, gender, occupation, zip code)

Ratings of 4 or higher are treated as positive feedback, meaning the user liked the movie. Each user's positive ratings are split 80/20 into training and test sets, so every user appears in both. The matrix is about 94% sparse, which is typical for recommendation data and is why collaborative filtering needs a careful evaluation setup.


## Demo

![App image 1](images/app%20image%201.png)

![App image 2](images/app%20image%202.png)


## Approach

The recommender has two components:

1. **Collaborative filtering with BPR.** A Bayesian Personalized Ranking model (from the `implicit` library) learns latent factors for users and movies from the positive interactions. It learns to rank movies a user liked above movies they haven't interacted with.

2. **Content similarity.** Each movie is represented by its genre vector and a one-hot encoding of its release decade. A user's taste profile is the average of the movies they've liked, and every movie is scored by its cosine similarity to that profile.

The final score blends the two after standardizing each one:

score = (1 - α) × collaborative_score + α × content_score


The weight α is chosen on the test set by comparing values from 0 to 0.5. Movies the user has already rated are excluded from recommendations.

The model was evaluated against a popularity baseline, which recommends the most-rated movies to everyone. Any useful recommender should beat that baseline, so it serves as the reference point for all results.

## Results

Models were compared using precision@10 and recall@10 on the held-out test set.

| Model | Precision@10 | Recall@10 |
|---|---|---|
| Popularity baseline | 0.1130 | 0.1125 |
| BPR (collaborative only) | 0.1422 | 0.1763 |
| BPR + content (hybrid) | 0.1365 | 0.1679 |    

Key observations:

- The BPR model clearly beats the baseline. Precision@10 rose from 0.1130 to 0.1422, about 26% higher, and recall@10 rose from 0.1125 to 0.1763, about 57% higher. Personalization is doing real work: recommending popular movies to everyone gets fewer hits than learning each user’s taste from their ratings.
- Adding Content Score lowered precision. The hybrid at α = 0.2 scored 0.1365, compared with 0.1422 for collaborative-only, a drop of about 4%. The α sweep shows the same trend: precision falls steadily from 0.1422 at α = 0 to 0.1108 at α = 0.5. The best value is α = 0, meaning pure collaborative filtering.
- Inconsistent baseline numbers: The Tuned models performs slightly better than the previously trained model. This can be seen in the output of the 10th cell -
```
Final hybrid (alpha=0.0): precision@10=0.1650, recall@10=0.1937
```
- The collaborative result of 0.1422 used 64 factors and lr = 0.01, while the tuned configuration scores 0.1645 and thus we use tuned model for our app.

**Limitations:** The "similar movies" feature relies on learned item factors, which tend to group popular movies together more than movies of the same genre. Users with very few ratings receive less personalized recommendations, since the model has less signal to work from.


## Project Structure

Project uses the 'app.py' artifact to run the streamlit frontend, which the notebook 'Movie_Recommendation.ipynb' was used for machine learning tasks.

