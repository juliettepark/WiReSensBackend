"""
Model to predict the color of a stress ball using tactile and handpose data on a pinch action.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/pinch_labeled_results.csv')
tactile_handpose_data = df.drop(columns=['label'])
labels = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tactile_handpose_data, labels, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")