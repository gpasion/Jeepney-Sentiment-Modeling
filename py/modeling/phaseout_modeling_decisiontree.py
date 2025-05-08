import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("/Users/gpasion/Documents/INST414/sentiments.csv")

# Vectorize the tweet text
vectorizer = TfidfVectorizer(max_features=500)
X_text = vectorizer.fit_transform(df['translated'])

# Combine with other features
X = pd.concat([
    pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out()),
    df[['likeCount', 'Relevant']].reset_index(drop=True)
], axis=1)

y = df['sentiment']  # Target variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Train the decision tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
plt.show()
