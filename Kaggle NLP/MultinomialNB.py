import spacy
from spacy.lang.en import STOP_WORDS
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv("train.csv")
# print(df.head())
# exit()
# Preprocess the data
df["text"] = df["text"].apply(lambda x: " ".join([word for word in x.lower().split() if word not in STOP_WORDS]))

# Vectorize the text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df["target"], test_size=0.1)

# Train the model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))