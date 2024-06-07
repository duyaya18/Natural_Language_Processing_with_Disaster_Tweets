import numpy as np
import pandas as pd
pd.set_option("display.max_colwidth", None)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

train_df = pd.read_csv("../input/nlp-getting-started/train.csv")
test_df = pd.read_csv("../input/nlp-getting-started/test.csv")
train_df

train_df.info()

train_df.isnull().sum()

train_df["target"].hist()

# location with text
train_df["location_text"] = [x[1] if pd.isna(x[0]) else (x[0]+" "+x[1]) for x in train_df[["location", "text"]].values]
test_df["location_text"] = [x[1] if pd.isna(x[0]) else (x[0]+" "+x[1]) for x in test_df[["location", "text"]].values]


##############################

import re
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")

lemmatizer = WordNetLemmatizer()

def clean_data(tweet):
    tweet = re.sub("[@&]\w*", "", tweet)
    tweet = re.sub("https?:\S*", "", tweet)
    tweet = re.sub("[^A-Za-z#]", " ", tweet)
    tweet = tweet.lower()
    tweet = [lemmatizer.lemmatize(word) for word in tweet.split() if word not in stopwords.words("english")]
    tweet = " ".join(tweet)

    return tweet

from nltk.corpus import stopwords

# add clean text column
train_df["clean_text"] = train_df["location_text"].apply(clean_data)
test_df["clean_text"] = test_df["location_text"].apply(clean_data)

train_df[['location_text', 'clean_text']]

y = train_df["target"].values
X = train_df["clean_text"].values.astype("U")    # "U" for Unicode string

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_val.shape)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_trans = TfidfTransformer()
X_train_tfidf = tfidf_trans.fit_transform(X_train_count)
X_val_tfidf = tfidf_trans.transform(X_val_count)

print(X_train_tfidf.shape, X_val_tfidf.shape)


######################

from sklearn.ensemble import StackingClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

xgb_clf = xgb.XGBClassifier(random_state=42)
log_reg = LogisticRegression(solver="newton-cg", random_state=42)
mnb = MultinomialNB()
sgd_clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)


estimators = [('xgb_clf', xgb_clf), ('sgd_clf', sgd_clf), ('mnb', mnb)]
final_estimator = log_reg

stacking_clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
stacking_clf.fit(X_train_tfidf, y_train)

scores = cross_val_score(stacking_clf, X_train_tfidf, y_train, scoring="accuracy", cv=5)

print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard deviation:", scores.std())

y_pred = stacking_clf.predict(X_val_tfidf)
cm = confusion_matrix(y_val, y_pred)

print(cm)
print(classification_report(y_val, y_pred))

X_test = test_df["clean_text"].values.astype("U")    # "U" for Unicode string

# count vectorization
X_test_count = count_vect.transform(X_test)

print(X_test_count.shape)

# TF*IDF transformation
X_test_tfidf = tfidf_trans.transform(X_test_count)

print(X_test_tfidf.shape)

predictions = stacking_clf.predict(X_test_tfidf)
predictions

output = pd.DataFrame({'id': test_df.id, 'target': predictions})
output.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")