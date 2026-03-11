"""
Model to predict the color of a stress ball using tactile and handpose data on a pinch action.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Feature columns (order must match streamHandposeAndGlove.PINCH_FEATURE_COLUMNS at predict time)
FEATURE_COLUMNS = [
    "index_avg", "index_min", "index_max",
    "thumb_avg", "thumb_min", "thumb_max",
    "finger_tip_avg", "finger_tip_min", "finger_tip_max",
]

df = pd.read_csv('data/pinch_dough/pinch_dough_labeled_results.csv')
X = df[FEATURE_COLUMNS]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the model
filename = 'pinch_dough_model.joblib'
joblib.dump(model, filename)