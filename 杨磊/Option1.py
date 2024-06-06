import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report


df_train = pd.read_csv("../数据/train.csv")
df_test = pd.read_csv("../数据/test.csv")

texts = df_train['text']
labels = df_train['target']
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

pipeline = make_pipeline(
    TfidfVectorizer(stop_words=stopwords.words('english')),
    MultinomialNB()
)

# 训练模型
pipeline.fit(X_train, y_train)
# 进行预测
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# 预测df_test
pre = df_test['text']
pre = pipeline.predict(pre)
submission = pd.DataFrame({
    'id' : df_test['id'],
    'target' : pre
})
# 输出结果
submission.to_csv("/预测数据/submission1",index=False)
print('成功')