import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words(‘english’))
df_text = pd.read_csv(‘fake_or_real_news.csv’, encoding=‘latin-1’)
df_text.columns = [‘id’, ‘title’, ‘text’, ‘label’]df_text.drop([‘id’, ‘title’], axis=1)
text = re.sub(r”http\S+|www\S+|https\S+”, ”, text, flags=re.MULTILINE)
    text = re.sub(r’\@\w+|\#’,”, text)
text = text.translate(str.maketrans(”, ”, string.punctuation))
tokens = word_tokenize(text)
words = [w for w in tokens if not w in stop_words]
from sklearn.feature_extraction.text import TfidfVectorizer

tf_vector = TfidfVectorizer(sublinear_tf=True)
tf_vector.fit(df_text[‘text’])
X_text = tf_vector.transform(df_text[‘text’].ravel())
y_values = np.array(df_text[‘label’].ravel())
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y_values)le.transform(y_values)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_text, y_values, test_size=0.15, random_state=120)

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver=‘lbfgs’)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(accuracy_score(y_test, y_predict))

