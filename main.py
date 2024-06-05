import numpy as np
import pandas as pd

# print("TensorFlow version:", tf.__version__)
# print("KerasNLP version:", keras_nlp.__version__)


df_train = pd.read_csv("数据/train.csv")
df_test = pd.read_csv("数据/test.csv")

print(df_train.head())