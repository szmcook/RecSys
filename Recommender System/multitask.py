# -*- coding: utf-8 -*-
"""Multitask.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1w51U36gPiS7T7KkfC-SbKzGGOn7FfU69
"""

# !pip install -q tensorflow-recommenders

import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from typing import Dict, Text

ratings = tf.data.experimental.CsvDataset(
    'ratings.csv',
    [tf.string, tf.string, tf.string, tf.int64, tf.int64],
    header=True
)

ratings = ratings.map(lambda u, i, n, m, r: {
    "user_id": u,
    "item_id": i,
    "name": n,
    "minutes": m,
    "rating": r,
})

recipes = tf.data.experimental.CsvDataset(
    'recipes.csv',
    [tf.string, tf.string, tf.int64, tf.string, tf.string, tf.string, tf.string, tf.int64, tf.string, tf.string, tf.string, tf.int64],
    header=True
)

recipes = recipes.map(lambda name,item_id,minutes,contributor_id,submitted,tags,nutrition,n_steps,steps,description,ingredients,n_ingredients: name)

# Randomly shuffle data and split between train and test.
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

recipes_titles = recipes.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

# unique_recipe_titles = np.unique(np.concatenate(list(recipes_titles)))
# unique_user_ids = np.unique(np.concatenate(list(user_ids)))

item_name_lookup = tf.keras.layers.StringLookup()
item_name_lookup.adapt(ratings.map(lambda x: x["name"]))

user_id_lookup = tf.keras.layers.StringLookup()
user_id_lookup.adapt(ratings.map(lambda x: str(x["user_id"])))

class RecipesModel(tfrs.models.Model):

  def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
    # We take the loss weights in the constructor: this allows us to instantiate
    # several model objects with different loss weights.

    super().__init__()

    embedding_dimension = 32

    # User and item models.
    self.item_model = tf.keras.Sequential([
      item_name_lookup,
      tf.keras.layers.Embedding(item_name_lookup.vocabulary_size(), 32)
    ])
    self.user_model = tf.keras.Sequential([
        user_id_lookup,
        tf.keras.layers.Embedding(user_id_lookup.vocabulary_size(), 32),
    ])

    # A small model to take in user and item embeddings and predict ratings.
    # We can make this as complicated as we want as long as we output a scalar
    # as our prediction.
    self.rating_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    # The tasks.
    self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=recipes.batch(128).map(self.item_model)
        )
    )

    # The loss weights.
    self.rating_weight = rating_weight
    self.retrieval_weight = retrieval_weight

  def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])
    # And pick out the item features and pass them into the item model.
    item_embeddings = self.item_model(features["name"])
    # print(user_embeddings.shape, item_embeddings.shape)
    c = tf.stack([user_embeddings, item_embeddings], axis=0)
    r = self.rating_model(c)
    return (user_embeddings, item_embeddings, r)
    # return (
    #     user_embeddings,
    #     item_embeddings,
    #     # We apply the multi-layered rating model to a concatentation of
    #     # user and item embeddings.
    #     self.rating_model(
    #         tf.concat([user_embeddings, item_embeddings], axis=1)
    #     ),
    # )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

    ratings = features.pop("rating")

    user_embeddings, item_embeddings, rating_predictions = self(features)

    # We compute the loss for each task.
    rating_loss = self.rating_task(
        labels=ratings,
        predictions=rating_predictions,
    )
    retrieval_loss = self.retrieval_task(user_embeddings, item_embeddings)

    # And combine them using the loss weights.
    return (self.rating_weight * rating_loss
            + self.retrieval_weight * retrieval_loss)

model = RecipesModel(rating_weight=1.0, retrieval_weight=0.0)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

model.fit(cached_train, epochs=10)

model.save_weights("CF")

n, u, pr = model({"user_id":np.array("599450"), "name":np.array("honey roasted plums with thyme")})
print(pr)

model.load_weights("CF")

trained_name_embeddings, trained_user_embeddings, predicted_rating = model({
      "user_id": np.array("5"),
      "name": np.array("chicken tagine with apricots and spiced pine nuts")
  })
print("Predicted rating:")
print(predicted_rating)