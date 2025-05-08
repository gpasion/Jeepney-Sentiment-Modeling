import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load your data
df = pd.read_csv("/Users/gpasion/Documents/INST414/sentiments.csv")

# Vectorize the tweet text
vectorizer = TfidfVectorizer(max_features=500)
X_text = vectorizer.fit_transform(df['translated'])

# Combine with other features
X = pd.concat([
    pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out()),
    df[['likeCount', 'Relevant']].reset_index(drop=True)
], axis=1)

y = df['sentiment']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))
