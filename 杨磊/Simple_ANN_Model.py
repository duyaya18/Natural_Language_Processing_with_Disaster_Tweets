import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import math
import io

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix, accuracy_score

dataset_path = "../数据"

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

# 词向量化层
max_tokens = 10_000
ngrams = None
output_sequence_length = math.ceil(np.array([len(seq.split()) for seq in X_train]).mean())

text_vectorizer = layers.TextVectorization(
    max_tokens=max_tokens,
    ngrams=ngrams, # create groups of n-words
    output_sequence_length=output_sequence_length, # sequences of words length
)

text_vectorizer.adapt(X_train) # 训练

# 词嵌入层
text_embedding = layers.Embedding(input_dim=max_tokens,
                                 output_dim=128,
                                 input_length=output_sequence_length)

model_1 = tf.keras.Sequential([
    layers.Input([1, ], dtype=tf.string),

    text_vectorizer,
    text_embedding,
    layers.GlobalAveragePooling1D(),

    layers.Flatten(),
    layers.Dense(1, activation="sigmoid"),
])

model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

ann_history = model_1.fit(X_train, y_train, epochs=7, validation_data=(X_test, y_test))
pd.DataFrame(ann_history.history).plot()