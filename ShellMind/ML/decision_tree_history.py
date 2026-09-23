import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re

# Read the history file
with open('history.txt', 'r') as f:
    lines = f.readlines()

# Strip newline characters and remove empty lines
commands = [line.strip() for line in lines if line.strip()]

print(f"Total commands: {len(commands)}")
print(f"First 5 commands: {commands[:5]}")

# Feature extraction function
def extract_features(cmd):
    features = {}
    features['length'] = len(cmd)
    features['num_words'] = len(cmd.split())
    features['first_word'] = cmd.split()[0] if cmd.split() else ''
    # Check for certain keywords
    features['has_git'] = 1 if 'git' in cmd else 0
    features['has_python'] = 1 if 'python' in cmd else 0
    features['has_rustc'] = 1 if 'rustc' in cmd else 0
    features['has_dot_slash'] = 1 if './' in cmd else 0
    features['has_cd'] = 1 if 'cd' in cmd else 0
    features['has_brew'] = 1 if 'brew' in cmd else 0
    features['has_python3'] = 1 if 'python3' in cmd else 0
    features['has_pnpm'] = 1 if 'pnpm' in cmd else 0
    features['has_npm'] = 1 if 'npm' in cmd else 0
    features['has_cargo'] = 1 if 'cargo' in cmd else 0
    features['has_agy'] = 1 if 'agy' in cmd else 0
    return features

# Extract features for each command
feature_list = []
for cmd in commands:
    feature_list.append(extract_features(cmd))

# Convert to DataFrame
df_features = pd.DataFrame(feature_list)

# Prepare labels: next command
# We'll create a label for each command except the last one
labels = commands[1:]  # label for command at i is command at i+1
# We need to align: features for command at i, label for command at i+1
# So we take features[0:-1] and labels[0:] (which is commands[1:])
X = df_features.iloc[:-1]  # all but the last row
y = labels  # same length as X

print(f"Features shape: {X.shape}")
print(f"Number of labels: {len(y)}")

# Encode the labels (next command strings) to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Number of unique next commands: {len(label_encoder.classes_)}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Show classification report (optional, but might be huge due to many classes)
# We'll print only if the number of classes is manageable, otherwise skip
if len(label_encoder.classes_) < 20:
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
else:
    print("Too many classes to show classification report.")

# Feature importance
if hasattr(clf, 'feature_importances_'):
    feature_importances = pd.DataFrame(clf.feature_importances_, index=X.columns, columns=['importance'])
    print("Feature importances:")
    print(feature_importances.sort_values('importance', ascending=False))

# Example prediction: predict the next command after the last command in the history
if len(commands) >= 1:
    last_command = commands[-1]
    last_features = extract_features(last_command)
    last_df = pd.DataFrame([last_features])
    # Ensure the columns are in the same order as training
    last_df = last_df[X.columns]
    last_pred_encoded = clf.predict(last_df)
    last_pred_command = label_encoder.inverse_transform(last_pred_encoded)[0]
    print(f"\nLast command: {last_command}")
    print(f"Predicted next command: {last_pred_command}")
else:
    print("Not enough commands to make a prediction.")

# Suggestion for another algorithm
print("\n--- Suggestion ---")
if accuracy < 0.7:  # arbitrary threshold
    print("The decision tree accuracy is low. Consider trying:")
    print("1. Random Forest or Gradient Boosting for better performance.")
    print("2. Using a sequence model like a Recurrent Neural Network (RNN) or Transformer for next command prediction.")
    print("3. Using a Markov Chain model, which is simple and effective for such sequential data.")
else:
    print("The decision tree performed reasonably well. However, for sequence prediction, you might still consider:")
    print("1. Random Forest for potentially better accuracy.")
    print("2. Using n-gram models or language models for more context.")