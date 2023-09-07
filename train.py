import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, balanced_accuracy_score
import xgboost as xgb
import pickle


reviews = pd.read_csv('reviews.csv')
labels = pd.read_csv('labels.csv')

X = reviews['text']
y = labels['sentiment']

# Converting the Positive/Negative to 0/1 labels
mapping = {'Positive': 1, 'Negative': 0}
y = y.map(mapping)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

with open('vectorizer.pkl', 'wb') as model_file:
    pickle.dump(vectorizer, model_file)

xgb_classifier = xgb.XGBClassifier()
xgb_classifier.fit(X_train_vec, y_train)

with open('xgb_classifier.pkl', 'wb') as model_file:
    pickle.dump(xgb_classifier, model_file)

y_pred = xgb_classifier.predict(X_test_vec)

# Calculating balanced accuracy
balanced_acc = balanced_accuracy_score(y_test, y_pred)

# Printing the result
print("Balanced Accuracy:", balanced_acc)

# Additional metrics
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Macro-average F1 Score: {macro_f1}")
print("Classification Report:\n", report)
