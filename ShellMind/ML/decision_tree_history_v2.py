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

# Feature extraction function (returns a dictionary)
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

# We will predict the first word of the next command
# Prepare labels: first word of the next command
next_first_words = []
for i in range(len(commands)-1):
    next_cmd = commands[i+1]
    next_first_word = next_cmd.split()[0] if next_cmd.split() else ''
    next_first_words.append(next_first_word)

print(f"Number of next first words: {len(next_first_words)}")
print(f"Unique first words: {set(next_first_words)}")

# We'll use a window of the last N commands (including the current) to predict the next first word
N = 3  # window size

# We need to create samples where each sample is the features of commands [i-N+1, i] (inclusive)
# and the label is the first word of command i+1
# We'll start at index = N-1 (so that we have at least N commands from 0 to N-1) and go up to len(commands)-2
samples = []
labels = []
for i in range(N-1, len(commands)-1):
    # Collect features from i-N+1 to i (inclusive)
    window_features = []
    for j in range(i-N+1, i+1):
        # Convert the feature dict to a list in a fixed order
        feat_dict = feature_list[j]
        # We'll use the same order as the columns in df_features
        window_features.append([feat_dict[col] for col in df_features.columns])
    # Flatten the window features
    flattened = [item for sublist in window_features for item in sublist]
    samples.append(flattened)
    labels.append(next_first_words[i])  # because next_first_words[i] corresponds to command i+1

X = np.array(samples)
y = np.array(labels)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Encode the labels (first words) to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Number of unique first words: {len(label_encoder.classes_)}")

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

# Show classification report if number of classes is reasonable
if len(label_encoder.classes_) <= 10:
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
else:
    print("Too many classes to show classification report.")

# Feature importance (optional, but we have many features)
if hasattr(clf, 'feature_importances_'):
    # Create feature names for the flattened window
    feature_names = []
    for n in range(N):
        for col in df_features.columns:
            feature_names.append(f"t-{N-1-n}_{col}")  # t-2, t-1, t for N=3
    feature_importances = pd.DataFrame(clf.feature_importances_, index=feature_names, columns=['importance'])
    print("\nTop 10 feature importances:")
    print(feature_importances.sort_values('importance', ascending=False).head(10))

# Example prediction: predict the first word of the next command after the last command in the history
if len(commands) >= N:
    # Take the last N commands
    last_window_indices = range(len(commands)-N, len(commands))
    window_features = []
    for j in last_window_indices:
        feat_dict = feature_list[j]
        window_features.append([feat_dict[col] for col in df_features.columns])
    flattened = [item for sublist in window_features for item in sublist]
    X_example = np.array([flattened])
    y_pred_encoded = clf.predict(X_example)
    y_pred_first_word = label_encoder.inverse_transform(y_pred_encoded)[0]
    print(f"\nLast {N} commands: {commands[-N:]}")
    print(f"Predicted first word of next command: {y_pred_first_word}")
    # Also show what the actual next command would be if we had it (but we don't)
    # We can show the last command's next is unknown.
else:
    print(f"Not enough commands to form a window of size {N}.")

# Suggestion for another algorithm
print("\n--- Suggestion ---")
if accuracy < 0.6:  # arbitrary threshold
    print("The decision tree accuracy is moderate to low. Consider trying:")
    print("1. Random Forest or Gradient Boosting for better performance.")
    print("2. Using a simple Markov Chain model on the first words (which might work well for this sequential data).")
    print("3. Using a Recurrent Neural Network (RNN) if you have more data.")
else:
    print("The decision tree performed reasonably well. However, for sequence prediction of categorical labels, you might still consider:")
    print("1. Random Forest for potentially better accuracy.")
    print("2. Using n-gram models (like a Markov model) for interpretability and simplicity.")