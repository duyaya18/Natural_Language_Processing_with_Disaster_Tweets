import numpy as np
import pandas as pd
import tensorflow as tf
import keras_core as keras
import keras_nlp
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# print("TensorFlow version:", tf.__version__)
# print("KerasNLP version:", keras_nlp.__version__)


df_train = pd.read_csv("数据/train.csv")
df_test = pd.read_csv("数据/test.csv")

print(df_train.head())