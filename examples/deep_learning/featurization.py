import pprint

import tensorflow_datasets as tfds

ratings = tfds.load("movielens/100k-ratings", split="train")

for x in ratings.take(1).as_numpy_iterator():
  pprint.pprint(x)

import numpy as np
import tensorflow as tf

movie_title_lookup = tf.keras.layers.StringLookup()
movie_title_lookup.adapt(ratings.map(lambda x: x["movie_title"]))

print(f"Vocabulary: {movie_title_lookup.get_vocabulary()[:3]}")

movie_title_lookup(["Star Wars (1977)", "One Flew Over the Cuckoo's Nest (1975)"])

# We set up a large number of bins to reduce the chance of hash collisions.
num_hashing_bins = 200_000
movie_title_hashing = tf.keras.layers.Hashing(
    num_bins=num_hashing_bins
)

movie_title_hashing(["Star Wars (1977)", "One Flew Over the Cuckoo's Nest (1975)"])

movie_title_embedding = tf.keras.layers.Embedding(
    # Let's use the explicit vocabulary lookup.
    input_dim=movie_title_lookup.vocab_size(),
    output_dim=32
)

movie_title_model = tf.keras.Sequential([movie_title_lookup, movie_title_embedding])
movie_title_model(["Star Wars (1977)"])

user_id_lookup = tf.keras.layers.StringLookup()
user_id_lookup.adapt(ratings.map(lambda x: x["user_id"]))
user_id_embedding = tf.keras.layers.Embedding(user_id_lookup.vocab_size(), 32)
user_id_model = tf.keras.Sequential([user_id_lookup, user_id_embedding])

for x in ratings.take(3).as_numpy_iterator():
  print(f"Timestamp: {x['timestamp']}.")

timestamp_normalization = tf.keras.layers.Normalization(axis=None)
timestamp_normalization.adapt(ratings.map(lambda x: x["timestamp"]).batch(1024))

for x in ratings.take(3).as_numpy_iterator():
  print(f"Normalized timestamp: {timestamp_normalization(x['timestamp'])}.")

max_timestamp = ratings.map(lambda x: x["timestamp"]).reduce(
    tf.cast(0, tf.int64), tf.maximum).numpy().max()
min_timestamp = ratings.map(lambda x: x["timestamp"]).reduce(
    np.int64(1e9), tf.minimum).numpy().min()

timestamp_buckets = np.linspace(
    min_timestamp, max_timestamp, num=1000)

print(f"Buckets: {timestamp_buckets[:3]}")

timestamp_embedding_model = tf.keras.Sequential([
  tf.keras.layers.Discretization(timestamp_buckets.tolist()),
  tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32)
])

for timestamp in ratings.take(1).map(lambda x: x["timestamp"]).batch(1).as_numpy_iterator():
  print(f"Timestamp embedding: {timestamp_embedding_model(timestamp)}.")                                       



title_text = tf.keras.layers.TextVectorization()
title_text.adapt(ratings.map(lambda x: x["movie_title"]))

for row in ratings.batch(1).map(lambda x: x["movie_title"]).take(1):
  print(title_text(row))

title_text.get_vocabulary()[40:45]


class UserModel(tf.keras.Model):
  
  def __init__(self):
    super().__init__()

    self.user_embedding = tf.keras.Sequential([
        user_id_lookup,
        tf.keras.layers.Embedding(user_id_lookup.vocab_size(), 32),
    ])
    self.timestamp_embedding = tf.keras.Sequential([
      tf.keras.layers.Discretization(timestamp_buckets.tolist()),
      tf.keras.layers.Embedding(len(timestamp_buckets) + 2, 32)
    ])
    self.normalized_timestamp = tf.keras.layers.Normalization(
        axis=None
    )

  def call(self, inputs):

    # Take the input dictionary, pass it through each input layer,
    # and concatenate the result.
    return tf.concat([
        self.user_embedding(inputs["user_id"]),
        self.timestamp_embedding(inputs["timestamp"]),
        tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1))
    ], axis=1)


user_model = UserModel()

user_model.normalized_timestamp.adapt(
    ratings.map(lambda x: x["timestamp"]).batch(128))

for row in ratings.batch(1).take(1):
  print(f"Computed representations: {user_model(row)[0, :3]}")



class MovieModel(tf.keras.Model):
  
  def __init__(self):
    super().__init__()

    max_tokens = 10_000

    self.title_embedding = tf.keras.Sequential([
      movie_title_lookup,
      tf.keras.layers.Embedding(movie_title_lookup.vocab_size(), 32)
    ])
    self.title_text_embedding = tf.keras.Sequential([
      tf.keras.layers.TextVectorization(max_tokens=max_tokens),
      tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
      # We average the embedding of individual words to get one embedding vector
      # per title.
      tf.keras.layers.GlobalAveragePooling1D(),
    ])

  def call(self, inputs):
    return tf.concat([
        self.title_embedding(inputs["movie_title"]),
        self.title_text_embedding(inputs["movie_title"]),
    ], axis=1)


movie_model = MovieModel()

movie_model.title_text_embedding.layers[0].adapt(
    ratings.map(lambda x: x["movie_title"]))

for row in ratings.batch(1).take(1):
  print(f"Computed representations: {movie_model(row)[0, :3]}")