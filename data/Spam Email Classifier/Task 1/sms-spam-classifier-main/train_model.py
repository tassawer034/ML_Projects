import pandas as pd
import numpy as np
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

ps = PorterStemmer()

# download if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ✅ fix for punkt_tab issue
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def transform_text(text):
    text = str(text).lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    stops = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stops]
    return ' '.join(ps.stem(t) for t in tokens)


def load_data(
    path=r"C:\Users\aibad\Downloads\ARCH\Spam Email Classifier\sms-spam-classifier-main(1)\sms-spam-classifier-main\spam.csv"
):
    df = pd.read_csv(path, encoding='latin-1')
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
    df = df.dropna()
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df['clean'] = df['message'].apply(transform_text)
    return df


if __name__ == '__main__':
    df = load_data()

    X = df['clean']
    y = df['label_num']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ TfidfVectorizer handles both tokenizing + tf-idf
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tf = vectorizer.fit_transform(X_train)
    X_test_tf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tf, y_train)

    preds = model.predict(X_test_tf)
    print('Accuracy:', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    # ✅ Save artifacts
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print('✅ Saved vectorizer.pkl and model.pkl')
