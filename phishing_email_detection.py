import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample dataset
data = {
    "email": [
        "Win a free iPhone now",
        "Your account has been hacked",
        "Meeting scheduled for tomorrow",
        "Project submission deadline",
        "Claim your prize immediately",
        "Team meeting at 10 AM"
    ],
    "label": [
        "Phishing",
        "Phishing",
        "Safe",
        "Safe",
        "Phishing",
        "Safe"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["email"])
y = df["label"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test Model
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Sample Email Prediction
test_email = "Win cash prize now"

email_vector = vectorizer.transform([test_email])
prediction = model.predict(email_vector)

print("\nTest Email:", test_email)
print("Prediction:", prediction[0])
