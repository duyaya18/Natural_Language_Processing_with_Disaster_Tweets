dataset_path = "/kaggle/input/nlp-getting-started"
#%%
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import math
import io

from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

train_data = pd.read_csv(dataset_path + "/train.csv")
test_data = pd.read_csv(dataset_path + "/test.csv")

train_data = train_data.sample(frac=1, random_state=42)
test_data = test_data.sample(frac=1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    train_data["text"].to_numpy(),
    train_data["target"].to_numpy(),
    test_size=0.1,
    random_state=42,
)

max_tokens = 10_000
ngrams = None
output_sequence_length = math.ceil(np.array([len(seq.split()) for seq in X_train]).mean())

text_vectorizer = layers.TextVectorization(
    max_tokens=max_tokens,
    ngrams=ngrams, # create groups of n-words
    output_sequence_length=output_sequence_length, # sequences of words length
)

text_vectorizer.adapt(X_train) # 训练

text_embedding = layers.Embedding(input_dim=max_tokens,
                                 output_dim=128,
                                 input_length=output_sequence_length)

model_2 = tf.keras.Sequential([
    layers.Input([1,], dtype=tf.string),
    text_vectorizer,
    text_embedding,
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(128),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model_2.compile(loss="binary_crossentropy",
               optimizer=tf.keras.optimizers.Adam(),
               metrics=["accuracy"])

model_2.summary()

model_2_history = model_2.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

pd.DataFrame(model_2_history.history).plot()