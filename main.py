import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle


df = pd.read_csv("bank_intents.csv")

X_train, X_test, Y_train, Y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression()

model.fit(X_train_vec, Y_train)

with open("bank_model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)